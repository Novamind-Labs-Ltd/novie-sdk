//! Callback token: HMAC-SHA256 JWT.
//!
//! Byte-for-byte compatible with `novie_platform.gateway.callbacks.tokens` —
//! the platform mints with Python, agents mint test tokens with Rust, both
//! sides verify each other's tokens. The wire layout is locked by `PLATFORM_
//! CALLBACK_SPEC §3`.
//!
//! # Notes on JSON key order
//!
//! HS256 JWT is signed over the literal `header.payload` ASCII string, so the
//! payload byte sequence must be identical across implementations. We
//! therefore handcraft the payload JSON in field-declaration order matching
//! Python's `dict` insertion order. Do NOT switch to a generic
//! `serde_json::to_vec` on a `BTreeMap` — that would alphabetise keys and
//! break the signature.

use std::collections::BTreeMap;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::Utc;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;

use crate::error::{Error, Result};

const ISS: &str = "novie-platform";
const AUD: &str = "novie-agent-callback";
/// Pre-encoded `{"alg":"HS256","typ":"JWT"}` — saves recomputing every mint.
const HEADER_B64: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";

type HmacSha256 = Hmac<Sha256>;

/// All claims carried by a callback token.
///
/// The serialised wire format is **flat JSON** (not a struct serialise) so
/// callers don't need to worry about field renames; see `to_wire_json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallbackTokenClaims {
    // Standard JWT fields
    pub iss: String,
    pub aud: String,
    pub iat: i64,
    pub exp: i64,
    pub jti: String,

    // ExecutionContext projection (3-letter keys to keep tokens compact)
    pub req: String,
    pub ten: String,
    pub ws: String,
    pub sid: String,
    pub tid: String,
    pub pid: String,
    pub ptp: String,
    pub aid: String,
    pub stp: String,

    /// Pass-through claims (e.g. `{"scope": "agent-status"}`). Empty maps are
    /// omitted from the wire payload to match Python.
    pub trh: BTreeMap<String, Value>,
}

impl CallbackTokenClaims {
    /// Serialise to the canonical wire JSON used for signing AND for
    /// inspection. Key order is locked.
    fn to_wire_json(&self) -> String {
        // Build the JSON manually so the key order matches Python's dict
        // insertion order in `tokens.py::to_dict`.
        let mut s = String::with_capacity(384);
        s.push('{');
        push_str(&mut s, "iss", &self.iss);
        s.push(',');
        push_str(&mut s, "aud", &self.aud);
        s.push(',');
        push_int(&mut s, "iat", self.iat);
        s.push(',');
        push_int(&mut s, "exp", self.exp);
        s.push(',');
        push_str(&mut s, "jti", &self.jti);
        s.push(',');
        push_str(&mut s, "req", &self.req);
        s.push(',');
        push_str(&mut s, "ten", &self.ten);
        s.push(',');
        push_str(&mut s, "ws", &self.ws);
        s.push(',');
        push_str(&mut s, "sid", &self.sid);
        s.push(',');
        push_str(&mut s, "tid", &self.tid);
        s.push(',');
        push_str(&mut s, "pid", &self.pid);
        s.push(',');
        push_str(&mut s, "ptp", &self.ptp);
        s.push(',');
        push_str(&mut s, "aid", &self.aid);
        s.push(',');
        push_str(&mut s, "stp", &self.stp);
        if !self.trh.is_empty() {
            s.push(',');
            // Serialise the trh sub-object via serde_json so nested values
            // (numbers, booleans, nested maps) get correct JSON rendering.
            // Note: BTreeMap iterates in sorted order, which matches Python's
            // `dict(passthrough_claims or {})` only when the caller already
            // sorted; for `verify` correctness this doesn't matter (we
            // recompute the signature from the wire bytes during verify).
            // It DOES matter for byte-equality with Python mint output —
            // callers minting cross-language tokens should pre-canonicalise.
            s.push_str("\"trh\":");
            s.push_str(
                &serde_json::to_string(&self.trh).expect("BTreeMap<_,Value> always serialises"),
            );
        }
        s.push('}');
        s
    }
}

fn push_str(buf: &mut String, key: &str, val: &str) {
    buf.push('"');
    buf.push_str(key);
    buf.push_str("\":");
    buf.push_str(&serde_json::to_string(val).expect("string always serialises"));
}

fn push_int(buf: &mut String, key: &str, val: i64) {
    buf.push('"');
    buf.push_str(key);
    buf.push_str("\":");
    buf.push_str(&val.to_string());
}

/// Subset of `ExecutionContext` needed to mint a token. Decoupled from the
/// full Python ExecutionContext so users don't have to model unused fields.
#[derive(Debug, Clone)]
pub struct MintContext<'a> {
    pub request_id: &'a str,
    pub session_id: &'a str,
    pub thread_id: &'a str,
    pub tenant_id: &'a str,
    pub workspace_id: &'a str,
    pub principal_id: &'a str,
    pub principal_type: &'a str,
}

/// Mint a callback token. Mirrors `mint_callback_token` in `tokens.py`.
pub fn mint_callback_token(
    ctx: &MintContext<'_>,
    agent_id: &str,
    step_id: &str,
    secret: &[u8],
    ttl_seconds: i64,
    passthrough_claims: BTreeMap<String, Value>,
) -> Result<String> {
    if ttl_seconds <= 0 {
        return Err(Error::InvalidArgument(
            "ttl_seconds must be positive".into(),
        ));
    }
    if secret.is_empty() {
        return Err(Error::InvalidArgument("secret must be non-empty".into()));
    }

    let iat = Utc::now().timestamp();
    mint_with_clock(
        ctx,
        agent_id,
        step_id,
        secret,
        ttl_seconds,
        passthrough_claims,
        iat,
        None,
    )
}

/// Test-only mint that lets the caller pin `iat` and `jti`. Public because
/// integration tests across language boundaries need it; production code
/// should call `mint_callback_token`.
#[allow(clippy::too_many_arguments)]
pub fn mint_with_clock(
    ctx: &MintContext<'_>,
    agent_id: &str,
    step_id: &str,
    secret: &[u8],
    ttl_seconds: i64,
    passthrough_claims: BTreeMap<String, Value>,
    iat: i64,
    jti: Option<&str>,
) -> Result<String> {
    let claims = CallbackTokenClaims {
        iss: ISS.to_string(),
        aud: AUD.to_string(),
        iat,
        exp: iat + ttl_seconds,
        jti: jti
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()[..16].to_string()),
        req: ctx.request_id.to_string(),
        ten: ctx.tenant_id.to_string(),
        ws: ctx.workspace_id.to_string(),
        sid: ctx.session_id.to_string(),
        tid: ctx.thread_id.to_string(),
        pid: ctx.principal_id.to_string(),
        ptp: ctx.principal_type.to_string(),
        aid: agent_id.to_string(),
        stp: step_id.to_string(),
        trh: passthrough_claims,
    };

    let payload_json = claims.to_wire_json();
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{HEADER_B64}.{payload_b64}");
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts key of any length");
    mac.update(signing_input.as_bytes());
    let sig = mac.finalize().into_bytes();
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig);
    Ok(format!("{signing_input}.{sig_b64}"))
}

/// Verify a callback token and return its claims. Mirrors
/// `verify_callback_token` (constant-time signature check, header / iss / aud
/// / exp validation).
pub fn verify_callback_token(
    token: &str,
    secret: &[u8],
    leeway_seconds: i64,
) -> Result<CallbackTokenClaims> {
    verify_with_clock(token, secret, Utc::now().timestamp(), leeway_seconds)
}

/// Test-friendly verify with a pinned `now` clock.
pub fn verify_with_clock(
    token: &str,
    secret: &[u8],
    now: i64,
    leeway_seconds: i64,
) -> Result<CallbackTokenClaims> {
    if secret.is_empty() {
        return Err(Error::InvalidArgument("secret must be non-empty".into()));
    }
    if token.is_empty() {
        return Err(Error::TokenMalformed("empty token".into()));
    }

    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(Error::TokenMalformed("token must have 3 parts".into()));
    }
    let [header_b64, payload_b64, sig_b64] = [parts[0], parts[1], parts[2]];

    let signing_input = format!("{header_b64}.{payload_b64}");
    let provided = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|e| Error::TokenMalformed(format!("signature not valid base64url: {e}")))?;
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts key of any length");
    mac.update(signing_input.as_bytes());
    mac.verify_slice(&provided)
        .map_err(|_| Error::TokenMalformed("signature mismatch".into()))?;

    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|e| Error::TokenMalformed(format!("header not valid base64url: {e}")))?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| Error::TokenMalformed(format!("header not valid JSON: {e}")))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("HS256")
        || header.get("typ").and_then(|v| v.as_str()) != Some("JWT")
    {
        return Err(Error::TokenMalformed(format!(
            "unexpected header: {header}"
        )));
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|e| Error::TokenMalformed(format!("payload not valid base64url: {e}")))?;
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| Error::TokenMalformed(format!("payload not valid JSON: {e}")))?;
    let claims = claims_from_value(&payload)?;

    if claims.iss != ISS {
        return Err(Error::TokenMalformed(format!(
            "unexpected iss: {:?}",
            claims.iss
        )));
    }
    if claims.aud != AUD {
        return Err(Error::TokenMalformed(format!(
            "unexpected aud: {:?}",
            claims.aud
        )));
    }
    if now > claims.exp + leeway_seconds {
        return Err(Error::TokenExpired);
    }
    Ok(claims)
}

fn claims_from_value(v: &Value) -> Result<CallbackTokenClaims> {
    let obj = v
        .as_object()
        .ok_or_else(|| Error::TokenMalformed("payload is not a JSON object".into()))?;

    fn get_str(obj: &serde_json::Map<String, Value>, k: &str) -> Result<String> {
        obj.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| Error::TokenMalformed(format!("missing string claim {k:?}")))
    }
    fn get_int(obj: &serde_json::Map<String, Value>, k: &str) -> Result<i64> {
        obj.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Error::TokenMalformed(format!("missing integer claim {k:?}")))
    }

    let trh = obj
        .get("trh")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    Ok(CallbackTokenClaims {
        iss: get_str(obj, "iss")?,
        aud: get_str(obj, "aud")?,
        iat: get_int(obj, "iat")?,
        exp: get_int(obj, "exp")?,
        jti: get_str(obj, "jti")?,
        req: get_str(obj, "req")?,
        ten: get_str(obj, "ten")?,
        ws: get_str(obj, "ws")?,
        sid: get_str(obj, "sid")?,
        tid: get_str(obj, "tid")?,
        pid: get_str(obj, "pid")?,
        ptp: get_str(obj, "ptp")?,
        aid: get_str(obj, "aid")?,
        stp: get_str(obj, "stp")?,
        trh,
    })
}

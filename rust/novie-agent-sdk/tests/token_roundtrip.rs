//! Mint + verify token round-trips.
//!
//! These tests pin the wire layout (key order, header, padding-free base64url)
//! and the verify behaviour (expiry, signature mismatch, malformed parts).

use std::collections::BTreeMap;

use novie_agent_sdk::{Error, MintContext, mint_with_clock, verify_with_clock};
use serde_json::json;

const SECRET: &[u8] = b"unit-test-secret-do-not-use-in-prod";

fn ctx() -> MintContext<'static> {
    MintContext {
        request_id: "req-1",
        session_id: "sess-1",
        thread_id: "thr-1",
        tenant_id: "tnt-1",
        workspace_id: "ws-1",
        principal_id: "user-1",
        principal_type: "user",
    }
}

#[test]
fn mint_then_verify_returns_same_claims() {
    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        BTreeMap::new(),
        1_700_000_000,
        Some("jti-fixed-123"),
    )
    .unwrap();

    let parts: Vec<&str> = token.split('.').collect();
    assert_eq!(parts.len(), 3, "token must be header.payload.sig");
    assert_eq!(
        parts[0], "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "header must be the canonical HS256 JWT prefix"
    );

    let claims = verify_with_clock(&token, SECRET, 1_700_000_001, 0).unwrap();
    assert_eq!(claims.aid, "agent-x");
    assert_eq!(claims.stp, "step-y");
    assert_eq!(claims.iat, 1_700_000_000);
    assert_eq!(claims.exp, 1_700_000_060);
    assert_eq!(claims.jti, "jti-fixed-123");
    assert_eq!(claims.iss, "novie-platform");
    assert_eq!(claims.aud, "novie-agent-callback");
}

#[test]
fn passthrough_claims_round_trip() {
    let mut trh = BTreeMap::new();
    trh.insert("scope".into(), json!("agent-status"));
    trh.insert("max_calls".into(), json!(42));

    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        trh.clone(),
        1_700_000_000,
        Some("jti-1"),
    )
    .unwrap();
    let claims = verify_with_clock(&token, SECRET, 1_700_000_001, 0).unwrap();
    assert_eq!(claims.trh, trh);
}

#[test]
fn empty_passthrough_omits_trh_key() {
    // The wire payload should NOT contain "trh" when empty — Python omits it.
    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        BTreeMap::new(),
        1_700_000_000,
        Some("jti-1"),
    )
    .unwrap();
    let parts: Vec<&str> = token.split('.').collect();
    use base64::Engine;
    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .unwrap();
    let payload = std::str::from_utf8(&payload_bytes).unwrap();
    assert!(
        !payload.contains("\"trh\""),
        "empty trh must not appear in wire payload, got {payload}"
    );
}

#[test]
fn expired_token_is_rejected() {
    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        BTreeMap::new(),
        1_700_000_000,
        Some("jti-1"),
    )
    .unwrap();
    // 120s past the exp, leeway=0 → expired
    let err = verify_with_clock(&token, SECRET, 1_700_000_180, 0).unwrap_err();
    assert!(matches!(err, Error::TokenExpired), "got {err:?}");
}

#[test]
fn leeway_allows_recently_expired_tokens() {
    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        BTreeMap::new(),
        1_700_000_000,
        Some("jti-1"),
    )
    .unwrap();
    // 30s past exp, leeway=60 → still ok
    let claims = verify_with_clock(&token, SECRET, 1_700_000_090, 60).unwrap();
    assert_eq!(claims.aid, "agent-x");
}

#[test]
fn signature_mismatch_is_rejected() {
    let token = mint_with_clock(
        &ctx(),
        "agent-x",
        "step-y",
        SECRET,
        60,
        BTreeMap::new(),
        1_700_000_000,
        Some("jti-1"),
    )
    .unwrap();
    let err = verify_with_clock(&token, b"wrong-secret", 1_700_000_001, 0).unwrap_err();
    assert!(matches!(err, Error::TokenMalformed(_)), "got {err:?}");
}

#[test]
fn malformed_token_is_rejected() {
    let err = verify_with_clock("not-a-token", SECRET, 1_700_000_000, 0).unwrap_err();
    assert!(matches!(err, Error::TokenMalformed(_)), "got {err:?}");
}

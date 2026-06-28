//! Parser tests for `AgentInvokePayload` & `extract_call_scope`.

use novie_agent_sdk::{AgentInvokePayload, extract_call_scope};
use serde_json::json;

#[test]
fn parses_full_invoke_payload() {
    let raw = json!({
        "context": {},
        "inputs": {
            "agent_id": "agt-1",
            "__call_scope__": {
                "workspace_scope": "per_task",
                "tenant_id": "tnt-1",
                "workspace_id": "ws-1",
                "scope_key": "task-77",
                "credentials": {
                    "kind": "per_task",
                    "ttl_seconds": 600,
                },
            },
        },
        "platform_callback": {
            "base_url": "http://localhost:8000/internal/callbacks",
        },
        "agent_status_callback": {
            "url": "http://localhost:8000/internal/callbacks/agent-status",
            "token": "xyz.uvw.rst",
            "session_id": "sess-1",
            "thread_id": "thr-1",
        },
    });

    let payload = AgentInvokePayload::from_value(raw).unwrap();
    let pc = payload.platform_callback.as_ref().unwrap();
    assert_eq!(pc.base_url, "http://localhost:8000/internal/callbacks");
    assert_eq!(pc.version, "v1");

    let asc = payload.agent_status_callback.as_ref().unwrap();
    assert_eq!(asc.session_id.as_deref(), Some("sess-1"));
    assert_eq!(asc.thread_id.as_deref(), Some("thr-1"));

    assert_eq!(payload.agent_id(), Some("agt-1"));
}

#[test]
fn rejects_non_object_payload() {
    let err = AgentInvokePayload::from_value(json!([1, 2, 3])).unwrap_err();
    assert!(format!("{err:?}").contains("must be a JSON object"));
}

#[test]
fn rejects_invalid_callback_block() {
    let raw = json!({
        "platform_callback": {
            "base_url": "",
        }
    });
    let err = AgentInvokePayload::from_value(raw).unwrap_err();
    assert!(format!("{err:?}").contains("base_url"), "{err:?}");
}

#[test]
fn rejects_unknown_callback_version() {
    let raw = json!({
        "platform_callback": {
            "base_url": "http://x",
            "version": "v9",
        }
    });
    let err = AgentInvokePayload::from_value(raw).unwrap_err();
    assert!(format!("{err:?}").contains("v9"), "{err:?}");
}

#[test]
fn extract_call_scope_from_inputs_dunder() {
    let inputs = json!({
        "__call_scope__": {
            "workspace_scope": "per_session",
            "tenant_id": "t-1",
            "workspace_id": "w-1",
            "scope_key": "k",
        }
    });
    let scope = extract_call_scope(&inputs).unwrap();
    assert_eq!(scope.tenant_id, "t-1");
    assert_eq!(scope.workspace_id, "w-1");
}

#[test]
fn extract_call_scope_returns_none_when_missing() {
    let v = json!({});
    assert!(extract_call_scope(&v).is_none());
}

#[test]
fn extract_call_scope_silently_ignores_invalid_shape() {
    let v = json!({"call_scope": "not-an-object"});
    assert!(extract_call_scope(&v).is_none());
}

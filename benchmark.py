#!/usr/bin/env python3
"""
Credit Assignment for Agent Traces — Frontier Benchmark
=========================================================

Compares 5 credit assignment methods + consensus + cascade across 15 agent
traces (10 standard + 5 adversarial) using free OpenRouter models.

Novel contributions:
  - Reasoning Pattern Attribution (RPA): linguistic-signal-based blame
  - Self-consistency probing: variance under temperature as uncertainty proxy
  - Cascaded credit assignment: cheap→expensive method pipeline
  - Cross-model agreement as correctness signal
  - Variance estimation with 95% confidence intervals

Requirements:
    pip install httpx

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python benchmark.py --quick          # 2 traces, 1 model, smoke test
    python benchmark.py                  # Full benchmark (all traces, all models)
    python benchmark.py --phase 2        # With variance estimation (3 runs)
    python benchmark.py --phase 3        # Full frontier (self-consistency, cascade)

Author: Research prototype for DevRev AI team
"""

from __future__ import annotations
import asyncio
import json
import time
import sys
import os
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import httpx

# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "ox-alpha": {
        "id": "stealth/ox-alpha",
        "name": "Ox Alpha",
        "tier": "strong",
        "supports_json_mode": True,
    },
    "dots3": {
        "id": "dots-studio/dots-3-note-preview:free",
        "name": "Dots3 Note 280B",
        "tier": "mid",
        "supports_json_mode": False,
    },
}

RATE_LIMIT_RPM = 12
RATE_LIMIT_DELAY = 60 / RATE_LIMIT_RPM

CACHE_DIR = Path("./benchmark_cache")
RESULTS_DIR = Path("./benchmark_results")

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Step:
    id: int
    action: str
    reasoning: str
    observation: str
    is_fault: bool = False

@dataclass
class Trace:
    task_name: str
    task_description: str
    domain: str
    steps: list[Step]
    success: bool
    error_message: str
    fault_step_id: int
    is_adversarial: bool = False
    adversarial_type: str = ""
    num_steps: int = 0
    fault_position: str = ""

    def __post_init__(self):
        self.num_steps = len(self.steps)
        ratio = self.fault_step_id / self.num_steps if self.num_steps else 0
        if ratio <= 0.33:
            self.fault_position = "early"
        elif ratio <= 0.66:
            self.fault_position = "mid"
        else:
            self.fault_position = "late"

    def to_text(self, include_outcome: bool = False) -> str:
        lines = [f"Task: {self.task_description}\n"]
        for s in self.steps:
            lines.append(
                f"Step {s.id}: [{s.action}]\n"
                f"  Reasoning: {s.reasoning}\n"
                f"  Observation: {s.observation}"
            )
        if include_outcome:
            lines.append(f"\nOutcome: {'SUCCESS' if self.success else 'FAILED'}")
            if self.error_message:
                lines.append(f"Error: {self.error_message}")
        return "\n\n".join(lines)

@dataclass
class BlameResult:
    method: str
    model: str
    trace_name: str
    scores: list[float]
    predicted_culprit: int
    ground_truth: int
    correct: bool
    reasoning: str
    latency_ms: float
    run_id: int = 0
    raw_response: str = ""
    confidence: float = 0.0

# ============================================================
# 10 STANDARD TRACES + 5 ADVERSARIAL
# ============================================================

def create_traces() -> list[Trace]:
    standard = [
        Trace(
            task_name="cache_placement",
            task_description="Add a caching layer to the user service to reduce DB load",
            domain="backend", fault_step_id=3, success=False,
            error_message="3 tests failed: cached responses bypass service-layer validation. Stale permissions served.",
            steps=[
                Step(1, "Read user_service.py", "Understand existing architecture.",
                     "Found UserService with get_user(), update_user(), validate_user(). Service layer handles validation before DB calls."),
                Step(2, "Read database.py", "Check data access patterns.",
                     "Direct SQL queries. get_user_by_id() returns raw rows. No cache exists."),
                Step(3, "Decide: cache at HTTP handler level",
                     "Caching at handler is fastest — skip service layer entirely. Maximum performance.",
                     "Plan: intercept in HTTP handler, cache full responses by user ID.", is_fault=True),
                Step(4, "Create cache.py", "Simple in-memory cache.",
                     "Dict-based cache with LRU eviction, configurable max size."),
                Step(5, "Wire into api/handlers.py", "Cache-aside pattern.",
                     "GET checks cache first, POST invalidates."),
                Step(6, "Run tests", "Verify integration.",
                     "FAIL: test_role_change, test_deactivation, test_concurrent_update. Cached data bypasses validation."),
            ]
        ),
        Trace(
            task_name="oauth_implicit",
            task_description="Add OAuth2 login for third-party integrations",
            domain="security", fault_step_id=3, success=False,
            error_message="Security review failed: implicit grant exposes tokens in URLs, no refresh tokens, deprecated by OWASP.",
            steps=[
                Step(1, "Read auth module", "Understand current auth.",
                     "Session-based auth with bcrypt. No OAuth."),
                Step(2, "Research OAuth provider", "Check supported grants.",
                     "Provider supports authorization_code, implicit, client_credentials. Redirect URIs must be registered."),
                Step(3, "Choose implicit grant flow",
                     "Simpler — token in redirect URL, no server exchange. Less code.",
                     "Plan: redirect user, receive access_token in URL fragment, use directly.", is_fault=True),
                Step(4, "Implement redirect endpoint", "OAuth redirect handler.",
                     "Created /auth/oauth/start with response_type=token."),
                Step(5, "Implement callback", "Token extraction.",
                     "Client-side JS extracts access_token from URL fragment, sends to server."),
                Step(6, "Run security review", "Check against requirements.",
                     "FAIL: implicit flow exposes tokens in browser history/referrer. No refresh token. OWASP deprecated."),
            ]
        ),
        Trace(
            task_name="wrong_index",
            task_description="Add index to speed up slow user search query",
            domain="data", fault_step_id=3, success=False,
            error_message="B-tree index unused. @> array containment needs GIN, not B-tree.",
            steps=[
                Step(1, "Analyze slow query", "Identify the bottleneck.",
                     "SELECT * FROM users WHERE tags @> ARRAY['premium'] AND region='us-east'. 4.2s on 2M rows. Seq scan."),
                Step(2, "Check existing indexes", "What's already indexed.",
                     "PK on id, btree on email, btree on created_at. Tags is TEXT[] array type. No index on tags."),
                Step(3, "Create B-tree index on tags",
                     "B-tree is standard, most well-understood index type.",
                     "CREATE INDEX idx_users_tags ON users(tags);", is_fault=True),
                Step(4, "Run EXPLAIN ANALYZE", "Check if index is used.",
                     "Seq scan still on tags @> condition. B-tree does not support @> operator."),
                Step(5, "Try REINDEX and ANALYZE", "Maybe stats are stale.",
                     "No change. B-tree fundamentally incompatible with array containment."),
            ]
        ),
        Trace(
            task_name="exception_swallow",
            task_description="Improve error handling in payment pipeline",
            domain="backend", fault_step_id=2, success=False,
            error_message="Payment gateway timeout silently swallowed. Order marked completed without payment. $12K lost.",
            steps=[
                Step(1, "Audit error handling", "Map current error paths.",
                     "Bare try/except around gateway calls. Errors logged but execution continues. No retry or circuit breaker."),
                Step(2, "Catch and suppress gateway exceptions",
                     "Prevent crashes by wrapping all calls. Return default response on failure.",
                     "Wrapped charge_card(), capture_payment() in try/except. On exception: log, return {'status':'ok','fallback':True}.", is_fault=True),
                Step(3, "Add structured logging", "Machine-parseable error logs.",
                     "JSON logging with error_type, order_id, timestamp."),
                Step(4, "Add retry with backoff", "Retry transient failures.",
                     "3 retries, exponential backoff. Retry on timeout and 5xx."),
                Step(5, "Run integration tests", "Verify error handling.",
                     "FAIL: test_gateway_timeout passes silently. Exception caught, returns ok, order proceeds without payment."),
            ]
        ),
        Trace(
            task_name="rate_limit_granularity",
            task_description="Add rate limiting to public API to prevent abuse",
            domain="api", fault_step_id=3, success=False,
            error_message="Single user rotates across endpoints, stays under each limit, 307 req/min total.",
            steps=[
                Step(1, "Analyze traffic patterns", "Understand usage distribution.",
                     "50K req/min. Top 1% generate 40% of traffic. Abuse: user rotates /search, /autocomplete, /export to bypass endpoint limits."),
                Step(2, "Review options", "Evaluate strategies.",
                     "Options: per-user global, per-endpoint, per-user-per-endpoint. Redis vs in-memory."),
                Step(3, "Implement per-endpoint limits",
                     "Simplest to implement and reason about.",
                     "Plan: /search=100/min, /autocomplete=200/min, /export=10/min. Fixed window, in-memory.", is_fault=True),
                Step(4, "Build middleware", "Rate limit enforcement.",
                     "RateLimitMiddleware checks per-endpoint counters. 429 on excess."),
                Step(5, "Load test", "Simulate abuse.",
                     "FAIL: user rotates endpoints (99+199+9=307 req/min), all under individual limits."),
            ]
        ),
        Trace(
            task_name="dns_ttl_migration",
            task_description="Migrate production API to new infrastructure with zero downtime",
            domain="infra", fault_step_id=2, success=False,
            error_message="30% of users hit old servers for 2 hours after cutover. DNS TTL was 86400s (24h), not lowered before migration.",
            steps=[
                Step(1, "Document current infrastructure", "Map all services and dependencies.",
                     "API at api.example.com (A record -> old LB). DNS TTL=86400s. 3 app servers, 1 DB."),
                Step(2, "Plan cutover: update DNS A record to new LB IP",
                     "Simple DNS swap is cleanest. One change, instant cutover.",
                     "Plan: deploy new infra, verify, swap DNS A record from old to new LB IP. Total cutover: <5 min.", is_fault=True),
                Step(3, "Deploy new infrastructure", "Stand up new servers.",
                     "New LB + 3 app servers + DB replica deployed. Health checks passing."),
                Step(4, "Verify new infra with direct IP", "Test bypassing DNS.",
                     "All endpoints respond correctly via direct IP. Load test passes."),
                Step(5, "Execute DNS cutover", "Swap the A record.",
                     "Updated A record to new LB IP. Change visible in DNS within 2 min."),
                Step(6, "Monitor post-cutover", "Watch for issues.",
                     "30% of traffic still hitting old servers 2h later. Clients cached old A record with 24h TTL."),
            ]
        ),
        Trace(
            task_name="shared_state_workers",
            task_description="Add a job counter to the background worker pool for monitoring",
            domain="backend", fault_step_id=3, success=False,
            error_message="Race condition: counter drifts under load. 10 workers increment shared dict without locks. Count=8,412 after processing 10,000 jobs.",
            steps=[
                Step(1, "Read worker_pool.py", "Understand worker architecture.",
                     "10 workers in separate threads. Each pulls from Redis queue, processes job, acks. No shared state between workers."),
                Step(2, "Design counter approach", "Need a counter all workers can update.",
                     "Options: Redis INCR (atomic), shared dict with lock, database counter. Need real-time reads from monitoring endpoint."),
                Step(3, "Use shared Python dict for counter",
                     "Simplest approach — dict is in-memory, fast reads for the monitoring endpoint. Workers increment on job completion.",
                     "Plan: module-level dict {'processed':0, 'failed':0}. Each worker does counter['processed'] += 1 after job.", is_fault=True),
                Step(4, "Implement counter module", "Create the counter.",
                     "counter.py with get_counts() and increment(key) functions. Dict shared across threads."),
                Step(5, "Add /metrics endpoint", "Expose counter for monitoring.",
                     "Returns JSON of current counter values. Sub-millisecond response."),
                Step(6, "Load test with 10K jobs", "Verify counter accuracy.",
                     "FAIL: counter shows 8,412 processed after 10,000 jobs. Race condition: += is not atomic across threads."),
            ]
        ),
        Trace(
            task_name="recursive_serialization",
            task_description="Add JSON export endpoint for organization data with nested teams",
            domain="api", fault_step_id=4, success=False,
            error_message="RecursionError on /api/org/export. Team A reports to Team B which reports to Team A — circular reference.",
            steps=[
                Step(1, "Read org data model", "Understand the schema.",
                     "Org has teams. Each team has members, sub_teams, and parent_team. Teams can be nested arbitrarily deep."),
                Step(2, "Read existing serializer", "Check if there's a JSON serializer.",
                     "Basic to_dict() on User model. No serializer for Team or Org. No nested serialization."),
                Step(3, "Design recursive serializer",
                     "Teams are nested, so recurse: serialize team -> serialize sub_teams -> serialize their sub_teams.",
                     "Plan: team.to_dict() calls [t.to_dict() for t in self.sub_teams]. Natural recursion follows the tree."),
                Step(4, "Implement without cycle detection",
                     "Follow the recursive plan from step 3. Tree structures don't have cycles.",
                     "Org.to_dict() -> Team.to_dict() -> recurse sub_teams. No visited set or depth limit.", is_fault=True),
                Step(5, "Test with sample data", "Verify with small org.",
                     "Works: 3-level hierarchy exports correctly. Nested JSON looks right."),
                Step(6, "Test with production data snapshot", "Real data has more complex structures.",
                     "FAIL: RecursionError. Team 'Platform' -> parent 'Engineering' -> sub_team 'Platform'. Circular reference."),
            ]
        ),
        Trace(
            task_name="hardcoded_creds",
            task_description="Add Stripe payment integration to the checkout service",
            domain="security", fault_step_id=2, success=False,
            error_message="Security scan blocked deploy: Stripe secret key hardcoded in config.py, committed to git.",
            steps=[
                Step(1, "Read Stripe API docs", "Understand integration requirements.",
                     "Need: secret key (sk_live_...) for server-side, publishable key (pk_live_...) for client. Secret key must never be exposed."),
                Step(2, "Add keys to config.py",
                     "Store keys where the app can read them. Config file is the standard place for app settings.",
                     "Added STRIPE_SECRET_KEY='sk_live_abc123...' and STRIPE_PUB_KEY='pk_live_xyz...' to config.py.", is_fault=True),
                Step(3, "Implement checkout endpoint", "Build the payment flow.",
                     "Created /api/checkout that creates Stripe PaymentIntent using config.STRIPE_SECRET_KEY."),
                Step(4, "Add webhook handler", "Handle payment confirmations.",
                     "Created /webhooks/stripe with signature verification using config.STRIPE_WEBHOOK_SECRET."),
                Step(5, "Test payment flow", "End-to-end test.",
                     "Payment flow works in test mode. Stripe dashboard shows successful test charges."),
                Step(6, "CI/CD security scan", "Pre-deploy checks.",
                     "BLOCKED: Secret key detected in config.py. File is tracked in git. Key exposed in commit history."),
            ]
        ),
        Trace(
            task_name="unbounded_query",
            task_description="Build admin dashboard endpoint to list all user activity logs",
            domain="data", fault_step_id=3, success=False,
            error_message="OOM kill in production. Query returned 4.2M rows. No LIMIT, no pagination. Pod killed at 2GB memory.",
            steps=[
                Step(1, "Check activity_logs table", "Understand the data.",
                     "4.2M rows. Columns: user_id, action, timestamp, metadata (JSONB). ~200 bytes per row. Grows ~50K rows/day."),
                Step(2, "Design API endpoint", "Admin needs to see all logs with filters.",
                     "GET /admin/logs?user_id=X&action=Y&from=DATE&to=DATE. Return JSON array of matching logs."),
                Step(3, "Implement: SELECT * with optional WHERE filters",
                     "Straightforward query. Filters narrow results. Most admin queries will have filters, so result sets should be manageable.",
                     "SELECT * FROM activity_logs WHERE (filters). Return all matching rows as JSON array.", is_fault=True),
                Step(4, "Add optional date filters", "Let admins filter by time range.",
                     "Added from/to params. WHERE timestamp BETWEEN from AND to. Defaults to no filter if not provided."),
                Step(5, "Test with filtered queries", "Verify endpoint works.",
                     "Works: /admin/logs?user_id=123&from=2026-01-01 returns 847 rows. Fast, correct."),
                Step(6, "Admin hits endpoint without filters", "Unfiltered request in production.",
                     "GET /admin/logs (no params). Returns all 4.2M rows. Pod OOM killed at 2GB. Service down for 3 min."),
            ]
        ),
    ]

    adversarial = [
        # ADV 1: Confident wrong — low hedging, high actual blame. RPA should FAIL here.
        Trace(
            task_name="adv_confident_wrong",
            task_description="Implement distributed lock for inventory reservation across 3 microservices",
            domain="backend", fault_step_id=3, success=False, is_adversarial=True,
            adversarial_type="confident_wrong",
            error_message="Double-sell: two services reserved same inventory simultaneously. Redis SETNX without TTL causes deadlocks on crash.",
            steps=[
                Step(1, "Map inventory flow", "Trace the reservation path across services.",
                     "order-svc -> inventory-svc -> warehouse-svc. Each checks stock independently. No coordination."),
                Step(2, "Evaluate lock strategies", "Compare options.",
                     "Options: Redis SETNX, ZooKeeper, DB advisory locks, Redlock. Redis simplest, already in stack."),
                Step(3, "Implement Redis SETNX lock",
                     "Redis SETNX is the industry standard for distributed locks. Well-documented, battle-tested, minimal latency.",
                     "SETNX lock:inventory:{sku}. Acquire before reserve, release after commit. Clean, simple, correct.", is_fault=True),
                Step(4, "Add lock to reservation flow", "Wire lock into inventory-svc.",
                     "acquire_lock() before check_stock(). release_lock() in finally block."),
                Step(5, "Integration test with 2 services", "Verify exclusion.",
                     "Single-service test passes. Two concurrent requests properly serialized."),
                Step(6, "Load test with 3 services under contention", "Production-like scenario.",
                     "FAIL: Service crashes mid-lock, SETNX has no TTL, lock never released. Other services deadlock. Also: two near-simultaneous SETNX calls on different Redis replicas both succeed."),
            ]
        ),
        # ADV 2: Sin of omission — agent didn't do something it should have.
        Trace(
            task_name="adv_omission",
            task_description="Add file upload endpoint with virus scanning for user documents",
            domain="security", fault_step_id=3, success=False, is_adversarial=True,
            adversarial_type="omission",
            error_message="Malicious .exe uploaded as profile_pic.jpg. No MIME validation, no magic-byte check. Content-Type trusted from client header.",
            steps=[
                Step(1, "Read upload requirements", "Understand what files are allowed.",
                     "User docs: PDF, DOCX, images (JPG/PNG). Max 10MB. Must be virus-scanned before storage."),
                Step(2, "Implement multipart upload handler", "Parse incoming files.",
                     "Use multipart parser. Extract filename, content_type from headers, file bytes from body."),
                Step(3, "Validate file extension and size",
                     "Check filename ends with allowed extension. Check content-length <= 10MB.",
                     "if not filename.endswith(('.pdf','.docx','.jpg','.png')): reject. if len(data)>10MB: reject.", is_fault=True),
                Step(4, "Add ClamAV virus scan", "Scan file bytes.",
                     "Pipe file bytes to clamd socket. Check for FOUND response. Reject infected files."),
                Step(5, "Store in S3 with metadata", "Persist the file.",
                     "Upload to s3://docs/{user_id}/{uuid}.{ext}. Store metadata in DB."),
                Step(6, "Penetration test", "Security team tests the endpoint.",
                     "FAIL: Tester renamed malware.exe to report.pdf. Extension check passed. No magic-byte validation. Malware stored and served."),
            ]
        ),
        # ADV 3: Multi-fault interaction — two faults that individually might be fine.
        Trace(
            task_name="adv_multi_fault",
            task_description="Add real-time notifications via WebSocket with message persistence",
            domain="backend", fault_step_id=3, success=False, is_adversarial=True,
            adversarial_type="multi_fault",
            error_message="Messages lost during reconnect. In-memory queue drops on server restart (fault A). No client-side ack/replay from last-seen offset (fault B). Either fix alone would prevent loss.",
            steps=[
                Step(1, "Design notification architecture", "Plan the system.",
                     "WebSocket server pushes events. Need: connection management, message fan-out, persistence for offline users."),
                Step(2, "Implement WebSocket server with in-memory queue",
                     "Fast fan-out via in-memory broadcast. Queue messages per connection.",
                     "In-memory dict: {conn_id: [msg1, msg2, ...]}. On connect, drain queue. On disconnect, buffer.", is_fault=True),
                Step(3, "Skip client-side message acknowledgment",
                     "Server tracks delivery via connection state. Client reconnect replays from queue. Simple and clean.",
                     "No ack protocol. Server assumes delivered if WebSocket send() didn't raise. Client reconnects to same conn_id.", is_fault=True),
                Step(4, "Add PostgreSQL persistence for offline messages", "Durable storage.",
                     "INSERT notification on create. DELETE on delivery. Offline users get all undelivered on login."),
                Step(5, "Test with simulated disconnects", "Verify reconnection.",
                     "Client disconnects/reconnects: messages replayed from in-memory queue. Works."),
                Step(6, "Test with server restart under load", "Chaos test.",
                     "FAIL: Server restart clears in-memory queue. Client reconnects, no messages. DB has them but no replay offset — client doesn't know what it missed."),
            ]
        ),
        # ADV 4: Late fault — buried deep at step 6/7, confuses methods that expect early faults.
        Trace(
            task_name="adv_late_fault",
            task_description="Build CI/CD pipeline for microservice with automated canary deployment",
            domain="infra", fault_step_id=6, success=False, is_adversarial=True,
            adversarial_type="late_fault",
            error_message="Canary passed but full rollout crashed. Canary checked HTTP 200 only, missed 5x latency regression. 100% traffic to slow build caused cascade failure.",
            steps=[
                Step(1, "Define pipeline stages", "Map the CI/CD flow.",
                     "Build -> Test -> Stage -> Canary (5%) -> Rollout (100%). Each stage gates the next."),
                Step(2, "Implement build and test stages", "Containerize and test.",
                     "Docker build with layer caching. Run unit + integration tests. Fail-fast on test failure."),
                Step(3, "Add staging environment", "Pre-production validation.",
                     "Deploy to staging namespace. Run smoke tests against staging endpoints."),
                Step(4, "Implement canary deployment", "Progressive rollout.",
                     "Deploy new version to 5% of pods. Route 5% traffic via service mesh weight."),
                Step(5, "Add health check monitoring", "Watch canary health.",
                     "Monitor canary pod: CPU, memory, restart count. 5-minute observation window."),
                Step(6, "Define canary success criteria: HTTP 200 rate > 99%",
                     "Standard health check. If 99% of requests return 200, promote to full rollout.",
                     "Canary gate: if error_rate < 1% for 5 min, proceed to 100% rollout. Only checks status codes.", is_fault=True),
                Step(7, "Full rollout on canary pass", "100% traffic to new version.",
                     "Canary passed (0.3% error rate). Promoted to full rollout. All pods replaced."),
                Step(8, "Production monitoring", "Watch production metrics.",
                     "FAIL: p99 latency 12x baseline. New version has O(n^2) serializer. Canary didn't check latency. Cascade timeout across downstream services."),
            ]
        ),
        # ADV 5: Red herring hedging — step with high hedging is actually fine, fault is in confident step.
        Trace(
            task_name="adv_red_herring",
            task_description="Migrate user sessions from server-side to JWT-based authentication",
            domain="security", fault_step_id=4, success=False, is_adversarial=True,
            adversarial_type="red_herring_hedging",
            error_message="JWT contains user role in payload. Role changes not reflected until token expires (24h). Privilege escalation: demoted admin retains admin JWT.",
            steps=[
                Step(1, "Audit current session system", "Understand what to replace.",
                     "Redis-backed sessions. 30-min TTL. Stores user_id, role, permissions. ~50K active sessions."),
                Step(2, "Choose JWT library and algorithm",
                     "Maybe RS256 for asymmetric? Or HS256 is simpler... Let's go with RS256 to be safe, though it might be overkill for our scale. Could revisit later.",
                     "Selected PyJWT with RS256. Generated 2048-bit RSA keypair. Private key in vault, public key distributed."),
                Step(3, "Design token payload",
                     "Not entirely sure what belongs in the token vs what should be looked up. Could go either way on permissions granularity. Probably better to keep it minimal?",
                     "Payload: {sub: user_id, role: user_role, permissions: [...], exp: +24h, iat: now}. Include role for fast authz checks."),
                Step(4, "Set token expiry to 24 hours",
                     "24h expiry reduces re-authentication friction. Users stay logged in for a full workday. Clean UX.",
                     "JWT exp = iat + 86400. No refresh token rotation. Revocation via deny-list checked on sensitive endpoints only.", is_fault=True),
                Step(5, "Implement JWT middleware", "Verify and decode on each request.",
                     "Verify RS256 signature, check exp, extract claims. Attach user context to request."),
                Step(6, "Migration test with role changes", "Verify session behavior.",
                     "FAIL: Admin demoted to viewer. Old JWT still valid for 23h. Admin endpoints accessible. No token revocation on role change."),
            ]
        ),
    ]

    return standard + adversarial


# ============================================================
# API CLIENT — JSON MODE, RETRY, CACHING
# ============================================================

class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self.last_call = 0.0

    async def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

rate_limiter = RateLimiter(RATE_LIMIT_RPM)

def cache_key(model: str, prompt: str, temperature: float = 0.1, run_id: int = 0) -> str:
    h = hashlib.sha256(f"{model}:{prompt}:{temperature}:{run_id}".encode()).hexdigest()[:16]
    return h

async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model_id: str,
    prompt: str,
    temperature: float = 0.1,
    use_json_mode: bool = True,
    run_id: int = 0,
    retries: int = 6,
) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    ck = cache_key(model_id, prompt, temperature, run_id)
    cache_path = CACHE_DIR / f"{ck}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)["response"]

    for attempt in range(retries):
        await rate_limiter.wait()
        try:
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are an expert at debugging AI agent traces. You MUST respond with ONLY valid JSON. No markdown fences, no backticks, no prose before or after the JSON object."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4000,
                "temperature": temperature,
            }
            if use_json_mode:
                body["response_format"] = {"type": "json_object"}

            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/devrev/credit-assignment-bench",
                },
                json=body,
                timeout=180.0,
            )

            if resp.status_code == 429:
                wait = min(90, 8 * (2 ** attempt))
                print(f"    [429] waiting {wait}s...", end="", flush=True)
                await asyncio.sleep(wait)
                continue

            if resp.status_code in (502, 503, 504):
                wait = 10 * (attempt + 1)
                print(f"    [{resp.status_code}] retry in {wait}s...", end="", flush=True)
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                if "overload" in err_msg.lower() or "temporarily" in err_msg.lower():
                    wait = 15 * (attempt + 1)
                    print(f"    [overload] retry in {wait}s...", end="", flush=True)
                    await asyncio.sleep(wait)
                    continue
                raise ValueError(f"API error: {err_msg}")

            text = data["choices"][0]["message"]["content"]

            with open(cache_path, "w") as f:
                json.dump({"model": model_id, "response": text, "temperature": temperature}, f)
            return text

        except httpx.HTTPStatusError as e:
            if attempt < retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            raise
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            if attempt < retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Failed after {retries} retries")


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if "```" in cleaned:
        lines = cleaned.split("\n")
        in_fence = False
        json_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                json_lines.append(line)
        fenced = "\n".join(json_lines).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # Brace-matching: find the outermost balanced { ... }
    depth = 0
    start_idx = -1
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start_idx >= 0:
                try:
                    return json.loads(cleaned[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = -1

    # Last resort: first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {cleaned[:300]}...")


# ============================================================
# CREDIT ASSIGNMENT METHODS
# ============================================================

async def method_llm_judge(client, api_key, model_id, trace: Trace, *, use_json_mode=True, run_id=0, temperature=0.1) -> BlameResult:
    t0 = time.time()
    n = len(trace.steps)
    prompt = f"""Here is a failed AI agent execution trace:

{trace.to_text(include_outcome=True)}

Which single step CAUSED the failure? The step where the error appeared is not necessarily the cause — look for the upstream DECISION that made failure inevitable.

Respond with this JSON:
{{"culprit_step": <int 1-{n}>, "scores": [<float 0-1 for each of the {n} steps>], "confidence": <float 0-1>, "reasoning": "<one sentence>"}}"""

    raw = await call_openrouter(client, api_key, model_id, prompt, temperature=temperature, use_json_mode=use_json_mode, run_id=run_id)
    data = parse_json_response(raw)
    scores = [float(x) for x in data.get("scores", [0.0] * n)][:n]
    culprit = int(data.get("culprit_step", 1))
    conf = float(data.get("confidence", 0.5))

    return BlameResult(
        method="LLM Judge", model=model_id, trace_name=trace.task_name,
        scores=scores, predicted_culprit=culprit, ground_truth=trace.fault_step_id,
        correct=(culprit == trace.fault_step_id),
        reasoning=data.get("reasoning", ""), latency_ms=(time.time()-t0)*1000,
        raw_response=raw, run_id=run_id, confidence=conf,
    )


async def method_hindsight(client, api_key, model_id, trace: Trace, *, use_json_mode=True, run_id=0, temperature=0.1) -> BlameResult:
    t0 = time.time()
    n = len(trace.steps)
    prompt = f"""You are performing ROOT CAUSE ANALYSIS on this failed agent trace.

{trace.to_text(include_outcome=True)}

With hindsight knowledge of the failure mode, score each step's CAUSAL contribution.
The step that made the DECISION leading to failure gets highest blame — not the step that detected the error.

Respond with this JSON:
{{"culprit_step": <int 1-{n}>, "scores": [<float 0-1 for each of the {n} steps>], "confidence": <float 0-1>, "reasoning": "<one sentence>"}}"""

    raw = await call_openrouter(client, api_key, model_id, prompt, temperature=temperature, use_json_mode=use_json_mode, run_id=run_id)
    data = parse_json_response(raw)
    scores = [float(x) for x in data.get("scores", [0.0] * n)][:n]
    culprit = int(data.get("culprit_step", 1))
    conf = float(data.get("confidence", 0.5))

    return BlameResult(
        method="Hindsight Critic", model=model_id, trace_name=trace.task_name,
        scores=scores, predicted_culprit=culprit, ground_truth=trace.fault_step_id,
        correct=(culprit == trace.fault_step_id),
        reasoning=data.get("reasoning", ""), latency_ms=(time.time()-t0)*1000,
        raw_response=raw, run_id=run_id, confidence=conf,
    )


async def method_prm(client, api_key, model_id, trace: Trace, *, use_json_mode=True, run_id=0, temperature=0.1) -> BlameResult:
    t0 = time.time()
    n = len(trace.steps)
    prompt = f"""You are a process reward model. For each step, estimate P(task_success | state after this step).
This is forward-looking: at each step, imagine you only see steps up to that point.
Do NOT look at the outcome — estimate success probability based only on the decisions made so far.

{trace.to_text(include_outcome=False)}

A sharp DROP in probability between consecutive steps identifies the causal step.

Respond with this JSON:
{{"success_probs": [<float 0-1 for each of the {n} steps>], "biggest_drop_step": <int 1-{n}>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}}"""

    raw = await call_openrouter(client, api_key, model_id, prompt, temperature=temperature, use_json_mode=use_json_mode, run_id=run_id)
    data = parse_json_response(raw)
    probs = [float(x) for x in data.get("success_probs", [0.5] * n)][:n]

    drops = []
    for i in range(n):
        if i == 0:
            drops.append(max(0, 0.8 - probs[i]))
        else:
            drops.append(max(0, probs[i-1] - probs[i]))
    max_d = max(drops) if max(drops) > 0 else 1.0
    scores = [d / max_d for d in drops]
    culprit = int(data.get("biggest_drop_step", drops.index(max(drops)) + 1))

    return BlameResult(
        method="Process Reward Model", model=model_id, trace_name=trace.task_name,
        scores=scores, predicted_culprit=culprit, ground_truth=trace.fault_step_id,
        correct=(culprit == trace.fault_step_id),
        reasoning=data.get("reasoning", ""), latency_ms=(time.time()-t0)*1000,
        raw_response=raw, run_id=run_id, confidence=float(data.get("confidence", 0.5)),
    )


async def method_counterfactual(client, api_key, model_id, trace: Trace, *, use_json_mode=True, run_id=0, temperature=0.1) -> BlameResult:
    t0 = time.time()
    n = len(trace.steps)
    prompt = f"""You are performing COUNTERFACTUAL ANALYSIS on a failed agent trace.

{trace.to_text(include_outcome=True)}

For each step: if the agent had made the BEST ALTERNATIVE decision at that step and continued normally, estimate P(task would succeed).
Steps where changing the decision most improves the outcome are the causal steps.

Respond with this JSON:
{{"counterfactual_success": [<float 0-1 for each of the {n} steps>], "most_impactful_step": <int 1-{n}>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}}"""

    raw = await call_openrouter(client, api_key, model_id, prompt, temperature=temperature, use_json_mode=use_json_mode, run_id=run_id)
    data = parse_json_response(raw)
    cf = [float(x) for x in data.get("counterfactual_success", [0.0] * n)][:n]
    max_cf = max(cf) if max(cf) > 0 else 1.0
    scores = [c / max_cf for c in cf]
    culprit = int(data.get("most_impactful_step", cf.index(max(cf)) + 1))

    return BlameResult(
        method="Counterfactual Replay", model=model_id, trace_name=trace.task_name,
        scores=scores, predicted_culprit=culprit, ground_truth=trace.fault_step_id,
        correct=(culprit == trace.fault_step_id),
        reasoning=data.get("reasoning", ""), latency_ms=(time.time()-t0)*1000,
        raw_response=raw, run_id=run_id, confidence=float(data.get("confidence", 0.5)),
    )


async def method_rpa(client, api_key, model_id, trace: Trace, *, use_json_mode=True, run_id=0, temperature=0.1) -> BlameResult:
    t0 = time.time()
    n = len(trace.steps)
    prompt = f"""Analyze the REASONING TEXT at each step for linguistic signals that predict decision importance.
This is about the TEXTURE of the reasoning, NOT whether decisions were correct.
Do NOT use the outcome to judge — analyze reasoning text ONLY.

{trace.to_text(include_outcome=False)}

For each step, extract these three signals:
1. HEDGING (0.0-1.0): Uncertainty markers. Words like "could","might","simplest","should be fine", comparative justifications, dismissive qualifiers = high hedging. Pure mechanical action with no decision = 0.0.
2. CONFLICT (0.0-1.0): Does the action contradict information from EARLIER observations? Cross-reference the reasoning with prior observation text. 0 = fully consistent, 1 = directly contradicts earlier evidence.
3. COMMITMENT (0.0-1.0): How much does this step constrain future steps? Architectural/design decisions = high. Reading a file or running a test = low.

Respond with this JSON:
{{"signals": [<for each of the {n} steps: {{"hedging": <float>, "conflict": <float>, "commitment": <float>}}>], "flagged_step": <int 1-{n}>, "confidence": <float 0-1>, "reasoning": "<one sentence>"}}"""

    raw = await call_openrouter(client, api_key, model_id, prompt, temperature=temperature, use_json_mode=use_json_mode, run_id=run_id)
    data = parse_json_response(raw)
    signals = data.get("signals", [{"hedging":0.1,"conflict":0.1,"commitment":0.1}]*n)

    raw_scores = []
    for s in signals[:n]:
        h = float(s.get("hedging", 0))
        c = float(s.get("conflict", 0))
        k = float(s.get("commitment", 0))
        score = (0.3*h + 0.35*c + 0.35*k) * (1 + c*k)
        raw_scores.append(score)

    max_s = max(raw_scores) if max(raw_scores) > 0 else 1.0
    scores = [s / max_s for s in raw_scores]
    culprit = int(data.get("flagged_step", raw_scores.index(max(raw_scores)) + 1))

    return BlameResult(
        method="RPA (Novel)", model=model_id, trace_name=trace.task_name,
        scores=scores, predicted_culprit=culprit, ground_truth=trace.fault_step_id,
        correct=(culprit == trace.fault_step_id),
        reasoning=data.get("reasoning", ""), latency_ms=(time.time()-t0)*1000,
        raw_response=raw, run_id=run_id, confidence=float(data.get("confidence", 0.5)),
    )


ALL_METHODS = {
    "judge": method_llm_judge,
    "hindsight": method_hindsight,
    "prm": method_prm,
    "counterfactual": method_counterfactual,
    "rpa": method_rpa,
}

METHOD_NAMES = {
    "judge": "LLM Judge",
    "hindsight": "Hindsight Critic",
    "prm": "Process Reward Model",
    "counterfactual": "Counterfactual Replay",
    "rpa": "RPA (Novel)",
}


# ============================================================
# STATISTICAL UTILITIES
# ============================================================

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))

def mrr(results: list[BlameResult]) -> float:
    rrs = []
    for r in results:
        if not r.scores:
            rrs.append(0.0)
            continue
        indexed = sorted(enumerate(r.scores, 1), key=lambda x: -x[1])
        for rank, (sid, _) in enumerate(indexed, 1):
            if sid == r.ground_truth:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


# ============================================================
# REPORTING
# ============================================================

def compute_stats(results: list[BlameResult]) -> dict:
    if not results:
        return {}
    correct = sum(1 for r in results if r.correct)
    n = len(results)
    ranks = []
    for r in results:
        if not r.scores:
            ranks.append(len(r.scores) if r.scores is not None else 0)
            continue
        indexed = sorted(enumerate(r.scores, 1), key=lambda x: -x[1])
        rank = next((i+1 for i, (sid, _) in enumerate(indexed) if sid == r.ground_truth), len(r.scores))
        ranks.append(rank)
    top2 = 0
    for r in results:
        if not r.scores:
            continue
        indexed = sorted(enumerate(r.scores, 1), key=lambda x: -x[1])
        top2_set = {indexed[j][0] for j in range(min(2, len(indexed)))}
        if r.ground_truth in top2_set:
            top2 += 1
    lo, hi = wilson_ci(correct, n)
    return {
        "top1_accuracy": correct / n,
        "top1_ci_lo": lo,
        "top1_ci_hi": hi,
        "top2_accuracy": top2 / n,
        "mrr": mrr(results),
        "mean_rank": sum(ranks) / n if ranks else 0,
        "correct": correct,
        "total": n,
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }


def print_report(all_results: list[BlameResult], traces: list[Trace], phase: int = 1):
    RESULTS_DIR.mkdir(exist_ok=True)
    std_traces = [t for t in traces if not t.is_adversarial]
    adv_traces = [t for t in traces if t.is_adversarial]

    print("\n" + "=" * 95)
    print(f"  CREDIT ASSIGNMENT BENCHMARK — {len(traces)} traces x {len(set(r.model for r in all_results))} models x 5 methods")
    print("=" * 95)

    methods = sorted(set(r.method for r in all_results))

    # --- Aggregate ---
    print(f"\n  {'METHOD':<25} {'Top-1':>7} {'95% CI':>14} {'Top-2':>7} {'MRR':>7} {'Latency':>9}")
    print("  " + "-" * 72)
    for method in methods:
        subset = [r for r in all_results if r.method == method]
        s = compute_stats(subset)
        if not s:
            continue
        print(f"  {method:<25} {s['top1_accuracy']:>6.0%} [{s['top1_ci_lo']:.0%}-{s['top1_ci_hi']:.0%}]"
              f" {s['top2_accuracy']:>6.0%} {s['mrr']:>6.3f} {s['avg_latency_ms']:>7.0f}ms")

    # --- Standard vs Adversarial ---
    if adv_traces:
        print(f"\n  {'METHOD':<25} {'Std Top-1':>10} {'Adv Top-1':>10} {'Adv Delta':>10}")
        print("  " + "-" * 58)
        for method in methods:
            std_sub = [r for r in all_results if r.method == method and r.trace_name in {t.task_name for t in std_traces}]
            adv_sub = [r for r in all_results if r.method == method and r.trace_name in {t.task_name for t in adv_traces}]
            ss = compute_stats(std_sub)
            sa = compute_stats(adv_sub)
            if ss and sa:
                delta = sa['top1_accuracy'] - ss['top1_accuracy']
                print(f"  {method:<25} {ss['top1_accuracy']:>9.0%} {sa['top1_accuracy']:>9.0%} {delta:>+9.0%}")

    # --- Per-model ---
    model_ids = sorted(set(r.model for r in all_results))
    if len(model_ids) > 1:
        print(f"\n  BREAKDOWN BY MODEL")
        print("  " + "-" * 80)
        for method in methods:
            print(f"\n  {method}:")
            for mid in model_ids:
                subset = [r for r in all_results if r.method == method and r.model == mid]
                if not subset:
                    continue
                s = compute_stats(subset)
                short = mid.split("/")[-1].split(":")[0][:25]
                print(f"    {short:<30} {s['top1_accuracy']:>6.0%} {s['top2_accuracy']:>6.0%} MRR={s['mrr']:.3f}")

    # --- Per-trace ---
    print(f"\n  PER-TRACE RESULTS")
    print("  " + "-" * 80)
    for trace in traces:
        tag = " [ADV]" if trace.is_adversarial else ""
        print(f"\n  {'>' if trace.is_adversarial else ' '} {trace.task_name}{tag} [{trace.domain}] "
              f"(GT=step {trace.fault_step_id}, pos={trace.fault_position}, {trace.num_steps} steps)")
        for method in methods:
            subset = [r for r in all_results if r.method == method and r.trace_name == trace.task_name]
            if not subset:
                continue
            votes = Counter(r.predicted_culprit for r in subset)
            majority = votes.most_common(1)[0][0]
            agreement = votes[majority] / len(subset)
            correct = majority == trace.fault_step_id
            icon = "+" if correct else "x"
            preds = ", ".join(f"{r.model.split('/')[-1].split(':')[0][:10]}={r.predicted_culprit}{'v' if r.correct else 'x'}" for r in subset)
            print(f"    {icon} {method:<25} -> step {majority} (agree={agreement:.0%})  [{preds}]")

    # --- RPA signal analysis ---
    rpa_results = [r for r in all_results if r.method == "RPA (Novel)" and r.raw_response]
    if rpa_results:
        print(f"\n  RPA SIGNAL ANALYSIS")
        print("  " + "-" * 80)
        for r in rpa_results[:10]:
            try:
                data = parse_json_response(r.raw_response)
                signals = data.get("signals", [])
                t = next((t for t in traces if t.task_name == r.trace_name), None)
                if not t or not signals:
                    continue
                fault_sig = signals[t.fault_step_id - 1] if t.fault_step_id <= len(signals) else {}
                print(f"    {r.trace_name:<24} fault_step={t.fault_step_id} "
                      f"H={fault_sig.get('hedging',0):.2f} C={fault_sig.get('conflict',0):.2f} "
                      f"K={fault_sig.get('commitment',0):.2f}  {'CORRECT' if r.correct else 'WRONG->'+str(r.predicted_culprit)}")
            except Exception:
                pass

    # --- Error analysis ---
    print(f"\n  ERROR ANALYSIS")
    print("  " + "-" * 80)
    for method in methods:
        errors = [r for r in all_results if r.method == method and not r.correct]
        if not errors:
            print(f"  {method}: No errors")
            continue
        print(f"\n  {method} ({len(errors)} errors):")
        for r in errors:
            short = r.model.split("/")[-1].split(":")[0][:15]
            print(f"    {r.trace_name:<24} [{short}] pred={r.predicted_culprit} GT={r.ground_truth}")
            if r.reasoning:
                print(f"      \"{r.reasoning[:100]}\"")

    # --- Key comparison ---
    print(f"\n  {'=' * 95}")
    print("  KEY COMPARISON: RPA (Novel) vs Counterfactual Replay")
    print("  " + "=" * 95)
    rpa_s = compute_stats([r for r in all_results if r.method == "RPA (Novel)"])
    cf_s = compute_stats([r for r in all_results if r.method == "Counterfactual Replay"])
    if rpa_s and cf_s:
        print(f"""
  Counterfactual:  {cf_s['top1_accuracy']:.0%} top-1 [{cf_s['top1_ci_lo']:.0%}-{cf_s['top1_ci_hi']:.0%}] | MRR {cf_s['mrr']:.3f} | {cf_s['avg_latency_ms']:.0f}ms
  RPA (Novel):     {rpa_s['top1_accuracy']:.0%} top-1 [{rpa_s['top1_ci_lo']:.0%}-{rpa_s['top1_ci_hi']:.0%}] | MRR {rpa_s['mrr']:.3f} | {rpa_s['avg_latency_ms']:.0f}ms
  {'RPA matches or exceeds Counterfactual.' if rpa_s['top1_accuracy'] >= cf_s['top1_accuracy'] else 'Counterfactual outperforms RPA — proxy does not fully capture causal structure.'}
""")

    # --- Domain + Position breakdowns ---
    print("  ACCURACY BY DOMAIN")
    print("  " + "-" * 80)
    domains = sorted(set(t.domain for t in traces))
    header = f"  {'Domain':<15}" + "".join(f"  {m[:12]:>12}" for m in methods)
    print(header)
    print("  " + "-" * (13 + 14 * len(methods)))
    for domain in domains:
        dt = {t.task_name for t in traces if t.domain == domain}
        print(f"  {domain:<15}", end="")
        for method in methods:
            sub = [r for r in all_results if r.method == method and r.trace_name in dt]
            if sub:
                acc = sum(1 for r in sub if r.correct) / len(sub)
                print(f"  {acc:>11.0%}", end="")
            else:
                print(f"  {'--':>12}", end="")
        print()

    print(f"\n  {'Position':<15}" + "".join(f"  {m[:12]:>12}" for m in methods))
    print("  " + "-" * (13 + 14 * len(methods)))
    for pos in ["early", "mid", "late"]:
        pt = {t.task_name for t in traces if t.fault_position == pos}
        print(f"  {pos:<15}", end="")
        for method in methods:
            sub = [r for r in all_results if r.method == method and r.trace_name in pt]
            if sub:
                acc = sum(1 for r in sub if r.correct) / len(sub)
                print(f"  {acc:>11.0%}", end="")
            else:
                print(f"  {'--':>12}", end="")
        print()

    # --- Consensus method ---
    print(f"\n  CONSENSUS METHOD (majority vote across 5 methods)")
    print("  " + "-" * 80)
    consensus_correct = 0
    consensus_total = 0
    for trace in traces:
        trace_results = [r for r in all_results if r.trace_name == trace.task_name and r.run_id == 0]
        if len(trace_results) < 3:
            continue
        votes = Counter(r.predicted_culprit for r in trace_results)
        consensus_pred = votes.most_common(1)[0][0]
        is_correct = consensus_pred == trace.fault_step_id
        consensus_correct += int(is_correct)
        consensus_total += 1
        icon = "+" if is_correct else "x"
        print(f"    {icon} {trace.task_name:<24} consensus=step {consensus_pred} (GT={trace.fault_step_id}) "
              f"votes={dict(votes)}")
    if consensus_total:
        lo, hi = wilson_ci(consensus_correct, consensus_total)
        print(f"\n    Consensus accuracy: {consensus_correct}/{consensus_total} = {consensus_correct/consensus_total:.0%} [{lo:.0%}-{hi:.0%}]")

    # --- Cross-model agreement ---
    model_ids_set = sorted(set(r.model for r in all_results))
    if len(model_ids_set) > 1:
        print(f"\n  CROSS-MODEL AGREEMENT AS SIGNAL")
        print("  " + "-" * 80)
        agree_correct = 0
        agree_total = 0
        disagree_correct = 0
        disagree_total = 0
        for trace in traces:
            for method in methods:
                sub = [r for r in all_results if r.method == method and r.trace_name == trace.task_name and r.run_id == 0]
                if len(sub) < 2:
                    continue
                preds = [r.predicted_culprit for r in sub]
                if len(set(preds)) == 1:
                    agree_total += 1
                    agree_correct += int(preds[0] == trace.fault_step_id)
                else:
                    disagree_total += 1
                    majority = Counter(preds).most_common(1)[0][0]
                    disagree_correct += int(majority == trace.fault_step_id)
        if agree_total:
            print(f"    When models AGREE:    {agree_correct}/{agree_total} = {agree_correct/agree_total:.0%} correct")
        if disagree_total:
            print(f"    When models DISAGREE: {disagree_correct}/{disagree_total} = {disagree_correct/disagree_total:.0%} correct")

    # --- Save JSON ---
    output = {
        "config": {"traces": len(traces), "models": model_ids_set, "methods": list(ALL_METHODS.keys()), "phase": phase},
        "aggregate": {method: compute_stats([r for r in all_results if r.method == method]) for method in methods},
        "all_results": [
            {"method": r.method, "model": r.model, "trace": r.trace_name,
             "predicted": r.predicted_culprit, "ground_truth": r.ground_truth,
             "correct": r.correct, "reasoning": r.reasoning, "scores": r.scores,
             "run_id": r.run_id, "confidence": r.confidence}
            for r in all_results
        ],
    }
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


# ============================================================
# VARIANCE ESTIMATION (Phase 2)
# ============================================================

def print_variance_report(all_results: list[BlameResult], traces: list[Trace]):
    print(f"\n  {'=' * 95}")
    print("  VARIANCE ESTIMATION (3 runs, temperature=0.3)")
    print("  " + "=" * 95)

    methods = sorted(set(r.method for r in all_results))
    print(f"\n  {'METHOD':<25} {'Mean Acc':>9} {'Std':>7} {'Stability':>10}")
    print("  " + "-" * 55)

    for method in methods:
        runs_acc = []
        for run_id in range(3):
            sub = [r for r in all_results if r.method == method and r.run_id == run_id]
            if sub:
                runs_acc.append(sum(1 for r in sub if r.correct) / len(sub))
        if runs_acc:
            mean_acc = sum(runs_acc) / len(runs_acc)
            std_acc = (sum((a - mean_acc)**2 for a in runs_acc) / len(runs_acc)) ** 0.5
            stability = 1.0 - std_acc
            print(f"  {method:<25} {mean_acc:>8.0%} {std_acc:>6.3f} {stability:>9.0%}")

    # Per-trace consistency
    print(f"\n  PER-TRACE CONSISTENCY (fraction of runs giving same answer)")
    print("  " + "-" * 80)
    for trace in traces:
        for method in methods:
            sub = [r for r in all_results if r.method == method and r.trace_name == trace.task_name]
            if len(sub) < 2:
                continue
            preds = [r.predicted_culprit for r in sub]
            mode = Counter(preds).most_common(1)[0][0]
            consistency = sum(1 for p in preds if p == mode) / len(preds)
            if consistency < 1.0:
                print(f"    {trace.task_name:<24} {method:<25} consistency={consistency:.0%} preds={preds}")


# ============================================================
# CASCADED CREDIT ASSIGNMENT (Phase 3)
# ============================================================

def print_cascade_report(all_results: list[BlameResult], traces: list[Trace]):
    print(f"\n  {'=' * 95}")
    print("  CASCADED CREDIT ASSIGNMENT")
    print("  " + "=" * 95)
    print("  Strategy: Run LLM Judge first. If confidence < threshold, escalate to expensive methods.")

    for threshold in [0.5, 0.7, 0.85]:
        judge_results = [r for r in all_results if r.method == "LLM Judge" and r.run_id == 0]
        escalation_methods = ["Hindsight Critic", "Counterfactual Replay", "RPA (Novel)"]

        cascade_correct = 0
        cascade_total = 0
        calls_saved = 0
        total_possible_calls = 0

        for trace in traces:
            jr = next((r for r in judge_results if r.trace_name == trace.task_name), None)
            if not jr:
                continue
            cascade_total += 1
            total_possible_calls += len(escalation_methods) + 1

            if jr.confidence >= threshold:
                cascade_correct += int(jr.correct)
                calls_saved += len(escalation_methods)
            else:
                escalation_results = [r for r in all_results if r.method in escalation_methods
                                      and r.trace_name == trace.task_name and r.run_id == 0]
                if escalation_results:
                    votes = Counter(r.predicted_culprit for r in escalation_results)
                    votes[jr.predicted_culprit] = votes.get(jr.predicted_culprit, 0) + 1
                    consensus = votes.most_common(1)[0][0]
                    cascade_correct += int(consensus == trace.fault_step_id)
                else:
                    cascade_correct += int(jr.correct)
                    calls_saved += len(escalation_methods)

        if cascade_total:
            pct_saved = calls_saved / total_possible_calls if total_possible_calls else 0
            lo, hi = wilson_ci(cascade_correct, cascade_total)
            print(f"\n  Threshold={threshold:.2f}: {cascade_correct}/{cascade_total}={cascade_correct/cascade_total:.0%} "
                  f"[{lo:.0%}-{hi:.0%}] | API calls saved: {pct_saved:.0%}")


# ============================================================
# MAIN
# ============================================================

async def run_benchmark(
    api_key: str,
    model_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    trace_indices: list[int] | None = None,
    phase: int = 1,
    include_adversarial: bool = True,
):
    all_traces = create_traces()
    if not include_adversarial:
        all_traces = [t for t in all_traces if not t.is_adversarial]
    if trace_indices is not None:
        all_traces = [all_traces[i] for i in trace_indices if i < len(all_traces)]

    models_to_run = MODELS if not model_keys else {k: MODELS[k] for k in model_keys if k in MODELS}
    methods_to_run = ALL_METHODS if not method_keys else {k: ALL_METHODS[k] for k in method_keys if k in ALL_METHODS}

    num_runs = 3 if phase >= 2 else 1
    total_calls = len(all_traces) * len(models_to_run) * len(methods_to_run) * num_runs
    est_min = total_calls * RATE_LIMIT_DELAY / 60

    print(f"Benchmark Phase {phase}: {len(all_traces)} traces x {len(models_to_run)} models x {len(methods_to_run)} methods x {num_runs} runs = {total_calls} calls")
    print(f"Estimated time: {est_min:.0f} min (at {RATE_LIMIT_RPM} req/min, cached calls instant)")
    print(f"Models: {', '.join(m['name'] for m in models_to_run.values())}")
    print()

    all_results: list[BlameResult] = []

    async with httpx.AsyncClient() as client:
        for run_id in range(num_runs):
            if num_runs > 1:
                print(f"\n--- Run {run_id+1}/{num_runs} {'(temp=0.1)' if run_id == 0 else '(temp=0.3)'} ---")
            temp = 0.1 if run_id == 0 else 0.3

            for mi, (model_key, model_info) in enumerate(models_to_run.items()):
                model_id = model_info["id"]
                model_name = model_info["name"]
                json_mode = model_info.get("supports_json_mode", False)

                for ti, trace in enumerate(all_traces):
                    for mk, method_fn in methods_to_run.items():
                        adv_tag = "[A]" if trace.is_adversarial else "   "
                        label = f"[{mi+1}/{len(models_to_run)}] {model_name[:12]:<12} {adv_tag} {trace.task_name:<24} {mk:<15}"
                        print(f"  {label}", end="", flush=True)

                        try:
                            result = await method_fn(client, api_key, model_id, trace,
                                                     use_json_mode=json_mode, run_id=run_id, temperature=temp)
                            icon = "+" if result.correct else "x"
                            print(f" {icon} ->step {result.predicted_culprit} ({result.latency_ms:.0f}ms)")
                            all_results.append(result)
                        except Exception as e:
                            err_msg = str(e)[:80]
                            print(f" ERR: {err_msg}")
                            all_results.append(BlameResult(
                                method=METHOD_NAMES.get(mk, mk), model=model_id,
                                trace_name=trace.task_name, scores=[], predicted_culprit=-1,
                                ground_truth=trace.fault_step_id, correct=False,
                                reasoning=f"Error: {err_msg}", latency_ms=0, run_id=run_id,
                            ))

    print_report(all_results, all_traces, phase)

    if phase >= 2:
        print_variance_report(all_results, all_traces)

    if phase >= 3:
        print_cascade_report(all_results, all_traces)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Assignment Benchmark — Frontier")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()))
    parser.add_argument("--methods", nargs="*", choices=list(ALL_METHODS.keys()))
    parser.add_argument("--traces", type=int, nargs="*")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="1=base, 2=+variance, 3=+cascade+self-consistency")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 2 traces, 1 model")
    parser.add_argument("--no-adversarial", action="store_true")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: Set OPENROUTER_API_KEY or pass --key")
        sys.exit(1)

    model_keys = args.models
    trace_indices = args.traces
    include_adv = not args.no_adversarial

    if args.quick:
        model_keys = ["ox-alpha"]
        trace_indices = [0, 3]
        include_adv = False

    asyncio.run(run_benchmark(api_key, model_keys, args.methods, trace_indices,
                              phase=args.phase, include_adversarial=include_adv))

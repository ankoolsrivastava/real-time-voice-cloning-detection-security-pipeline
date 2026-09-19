import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000/api";

type Result = {
  window_sequence?: number;
  prediction?: string;
  spoof_probability?: number;
  bonafide_probability?: number;

  prosody_probability?: number;
  prosody_reliability?: number;

  quality?: {
    score?: number;
    confidence_multiplier?: number;
    status?: string;
    rms_db?: number;
    peak?: number;
    clipping_ratio?: number;
    duration_sec?: number;
    active_ratio?: number;
  };

  risk?: {
    score?: number;
    level?: string;
    evidence_confidence?: number;
    status?: string;
    reasons?: string[];
  };

  temporal?: {
    current_risk?: number;
    accumulated_risk?: number;
    max_risk?: number;
    trend?: string;
    windows_seen?: number;
    high_windows?: number;
    critical_windows?: number;
    persistence?: number;
  };

  model?: {
    version?: string;
    device?: string;
  };
};

type Activity = {
  time: string;
  text: string;
  type: "ok" | "warn" | "info";
};

function App() {
  const [connected, setConnected] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");

  const [chunksSent, setChunksSent] = useState(0);
  const [lastSequence, setLastSequence] = useState<number | null>(null);
  const [activities, setActivities] = useState<Activity[]>([]);

  const audioContext = useRef<AudioContext | null>(null);
  const processor = useRef<ScriptProcessorNode | null>(null);
  const source = useRef<MediaStreamAudioSourceNode | null>(null);
  const stream = useRef<MediaStream | null>(null);

  const audioBuffer = useRef<number[]>([]);
  const sequence = useRef(0);
  const startTime = useRef<number>(0);
  const sending = useRef(false);

  const risk = result?.risk?.score ?? 0;
  const level = result?.risk?.level ?? "WAITING";
  const spoof = result?.spoof_probability ?? 0;

  const quality = result?.quality?.status ?? "WAITING";

  const modelVersion = result?.model?.version ?? "—";
  const modelDevice = result?.model?.device ?? "—";

  function addActivity(
    text: string,
    type: Activity["type"] = "ok",
  ) {
    const now = new Date();

    const time = now.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    setActivities((previous) => [
      { time, text, type },
      ...previous,
    ].slice(0, 8));
  }

  useEffect(() => {
    checkBackend();

    return () => {
      stopMicrophone();
    };
  }, []);

  async function checkBackend() {
    try {
      const response = await fetch(`${API}/health`);

      if (!response.ok) {
        throw new Error();
      }

      setConnected(true);
      setError("");
    } catch {
      setConnected(false);
      setError(
        "Backend unavailable. Start the VaaniX backend first.",
      );
    }
  }

  async function startSession() {
    try {
      setError("");
      setResult(null);
      setActivities([]);
      setChunksSent(0);
      setLastSequence(null);
      sequence.current = 0;

      addActivity("Creating backend session...", "info");

      const response = await fetch(`${API}/sessions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(
          `Session creation failed (${response.status})`,
        );
      }

      const data = await response.json();

      if (!data.session_id) {
        throw new Error(
          "Backend did not return a session ID.",
        );
      }

      const id = data.session_id;

      setSessionId(id);

      addActivity(
        `Session created: ${id.slice(0, 12)}...`,
      );

      await startMicrophone(id);

      setRunning(true);

      addActivity(
        "Microphone stream connected",
      );

      addActivity(
        "16 kHz mono audio capture active",
      );
    } catch (err) {
      setRunning(false);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to start VaaniX session.",
      );
    }
  }

  async function startMicrophone(id: string) {
    const mediaStream =
      await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

    stream.current = mediaStream;

    const context = new AudioContext({
      sampleRate: 16000,
    });

    await context.resume();

    audioContext.current = context;

    const micSource =
      context.createMediaStreamSource(mediaStream);

    const node = context.createScriptProcessor(
      16384,
      1,
      1,
    );

    source.current = micSource;
    processor.current = node;

    startTime.current = Date.now();

    node.onaudioprocess = (event) => {
      const input =
        event.inputBuffer.getChannelData(0);

      for (let i = 0; i < input.length; i++) {
        audioBuffer.current.push(input[i]);
      }

      if (
        audioBuffer.current.length >= 16000 &&
        !sending.current
      ) {
        const chunk =
          audioBuffer.current.splice(0, 16000);

        sendChunk(id, chunk);
      }
    };

    micSource.connect(node);
    node.connect(context.destination);
  }

  async function sendChunk(
    id: string,
    audio: number[],
  ) {
    sending.current = true;

    const currentSequence = sequence.current++;

    try {
      const now = Date.now();

      const timestampEnd =
        (now - startTime.current) / 1000;

      const timestampStart =
        Math.max(0, timestampEnd - 1);

      const response = await fetch(
        `${API}/sessions/${id}/chunks`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            audio,
            sample_rate: 16000,
            sequence_number: currentSequence,
            timestamp_start: timestampStart,
            timestamp_end: timestampEnd,
            received_at: Date.now() / 1000,
            packet_loss_before: 0,
            jitter_ms: 0,
          }),
        },
      );

      if (!response.ok) {
        const text = await response.text();

        throw new Error(
          `Chunk rejected (${response.status}): ${text}`,
        );
      }

      const data = await response.json();

      setChunksSent((value) => value + 1);
      setLastSequence(currentSequence);

      addActivity(
        `Audio chunk accepted • seq ${currentSequence}`,
      );

      if (data.result) {
        processResult(data.result);
      } else if (
        data.risk ||
        data.spoof_probability !== undefined ||
        data.prediction
      ) {
        processResult(data);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Audio chunk processing failed.",
      );

      addActivity(
        "Backend rejected audio chunk",
        "warn",
      );
    } finally {
      sending.current = false;
    }
  }

  function processResult(data: Result) {
    setResult(data);

    addActivity(
      "Analysis window processed by backend",
    );

    if (data.model?.version) {
      addActivity(
        `ML result received • ${data.model.version}`,
      );
    }

    if (data.quality?.status) {
      addActivity(
        `Signal quality assessed • ${data.quality.status}`,
      );
    }

    if (data.risk?.level) {
      const riskText =
        data.risk.level === "LOW"
          ? "Security policy: CONTINUE"
          : data.risk.level === "MEDIUM"
            ? "Security policy: VERIFY"
            : data.risk.level === "HIGH"
              ? "Security policy: MFA / VERIFY"
              : "Security policy: CALLBACK / ESCALATE";

      addActivity(
        riskText,
        data.risk.level === "LOW"
          ? "ok"
          : "warn",
      );
    }
  }

  function stopMicrophone() {
    processor.current?.disconnect();
    source.current?.disconnect();

    stream.current?.getTracks().forEach((track) => {
      track.stop();
    });

    audioContext.current?.close();

    processor.current = null;
    source.current = null;
    stream.current = null;
    audioContext.current = null;

    audioBuffer.current = [];
    sending.current = false;
  }

  async function stopSession() {
    stopMicrophone();

    if (sessionId) {
      try {
        await fetch(
          `${API}/sessions/${sessionId}`,
          {
            method: "DELETE",
          },
        );
      } catch {
        // Backend session may already be closed.
      }
    }

    setRunning(false);

    addActivity(
      "Session stopped",
      "info",
    );
  }

  function chooseAction(action: string) {
    addActivity(
      `Operator selected ${action}`,
      "info",
    );
  }

  const riskClass =
    level.toLowerCase();

  return (
    <div className="app">

      {/* HEADER */}

      <header className="topbar">
        <div>
          <div className="brand">
            VaaniX
          </div>

          <div className="subtitle">
            Voice Integrity Security Layer
          </div>
        </div>

        <div
          className={`status ${
            connected
              ? "online"
              : "offline"
          }`}
        >
          <span />
          {connected
            ? "BACKEND ONLINE"
            : "DISCONNECTED"}
        </div>
      </header>

      <main>

        {/* HERO */}

        <section className="hero">

          <div>
            <p className="eyebrow">
              REAL-TIME VOICE SECURITY
            </p>

            <h1>
              Detect suspicious voice evidence
              before trust becomes risk.
            </h1>

            <p className="heroText">
              VaaniX continuously evaluates
              voice integrity and converts
              detection evidence into an
              actionable security decision.
            </p>

            <div
              className={
                running
                  ? "liveMessage active"
                  : "liveMessage"
              }
            >
              <span>
                {running ? "●" : "○"}
              </span>

              {running
                ? "LIVE — microphone stream being analyzed"
                : "Ready to analyze voice"}
            </div>
          </div>

          {!running ? (
            <button
              className="primary"
              onClick={startSession}
              disabled={!connected}
            >
              START SESSION
            </button>
          ) : (
            <button
              className="stop"
              onClick={stopSession}
            >
              STOP SESSION
            </button>
          )}

        </section>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {/* LIVE PIPELINE */}

<section className="pipelineCard">

  <div className="sectionHeader">
    <div>
      <span className="sectionTag">
        LIVE PROCESSING
      </span>

      <h2>
        VaaniX Analysis Pipeline
      </h2>
    </div>

    <div className="pipelineMeta">
      {chunksSent} chunks transmitted
    </div>
  </div>

  <div className="pipeline">

    <PipelineStep
      number="01"
      title="INPUT"
      description="Live Voice / Call"
      status={
        running
          ? "RECEIVING"
          : "IDLE"
      }
      active={running}
    />

    <Arrow />

    <PipelineStep
      number="02"
      title="PREPARE"
      description="Streaming • 16 kHz • 10 sec"
      status={
        running
          ? "STREAMING"
          : "WAITING"
      }
      active={running}
    />

    <Arrow />

    <PipelineStep
      number="03"
      title="EVIDENCE"
      description="Acoustic • Prosody • Reliability"
      status={
        result
          ? "GENERATED"
          : running
            ? "ANALYZING"
            : "WAITING"
      }
      active={Boolean(result)}
    />

    <Arrow />

    <PipelineStep
      number="04"
      title="FUSION + RISK"
      description="Evidence Fusion • Dynamic 0–100"
      status={
        result
          ? `${Math.round(risk)} / 100`
          : "WAITING"
      }
      active={Boolean(result)}
    />

    <Arrow />

    <PipelineStep
      number="05"
      title="SECURITY ACTION"
      description="Continue • Verify • MFA • Callback"
      status={
        result?.risk?.level ??
        "WAITING"
      }
      active={Boolean(result)}
    />

  </div>

  <div className="systemLayer">

    <span className="systemLayerLabel">
      SYSTEM LAYER
    </span>

    <span>SESSION</span>
    <span>SEQUENCE</span>
    <span>TIMESTAMP</span>
    <span>PACKET LOSS</span>
    <span>JITTER</span>
    <span>LOGGING</span>

  </div>

</section>
        {/* MAIN DATA */}

        <section className="grid">

          {/* RISK */}

          <div className="card riskCard">

            <div className="cardTitle">
              VOICE INTEGRITY RISK
            </div>

            <div className="riskRing">
              <div>
                <strong>
                  {Math.round(risk)}
                </strong>

                <span>
                  /100
                </span>
              </div>
            </div>

            <div
              className={`riskLevel ${riskClass}`}
            >
              {level}
            </div>

            <div className="metric">
              <span>
                Spoof probability
              </span>

              <strong>
                {Math.round(
                  spoof * 100,
                )}
                %
              </strong>
            </div>

            <div className="metric">
              <span>
                Bonafide probability
              </span>

              <strong>
                {Math.round(
                  (result?.bonafide_probability ??
                    0) * 100,
                )}
                %
              </strong>
            </div>

          </div>

          {/* EVIDENCE */}

          <div className="card">

            <div className="cardTitle">
              MULTI-EVIDENCE ANALYSIS
            </div>

            <Evidence
              label="Acoustic / Spectral"
              value={spoof}
            />

            <Evidence
              label="Prosodic"
              value={
                result?.prosody_probability ??
                0
              }
            />

            <Evidence
              label="Temporal Risk"
              value={
                (result?.temporal
                  ?.accumulated_risk ??
                  risk) / 100
              }
            />

            <Evidence
              label="Evidence Confidence"
              value={
                result?.risk
                  ?.evidence_confidence ??
                0
              }
            />

            <div className="smallInfo">
              Prosody reliability:{" "}
              <strong>
                {result?.prosody_reliability !==
                undefined
                  ? result.prosody_reliability.toFixed(
                      2,
                    )
                  : "—"}
              </strong>
            </div>

          </div>

          {/* RELIABILITY */}

          <div className="card">

            <div className="cardTitle">
              SIGNAL RELIABILITY
            </div>

            <InfoRow
              label="Audio quality"
              value={quality}
            />

            <InfoRow
              label="Quality score"
              value={
                result?.quality?.score !==
                undefined
                  ? Math.round(
                      result.quality.score,
                    )
                  : "—"
              }
            />

            <InfoRow
              label="Confidence multiplier"
              value={
                result?.quality
                  ?.confidence_multiplier !==
                undefined
                  ? result.quality.confidence_multiplier.toFixed(
                      2,
                    )
                  : "—"
              }
            />

            <InfoRow
              label="Active speech"
              value={
                result?.quality
                  ?.active_ratio !==
                undefined
                  ? `${Math.round(
                      result.quality.active_ratio *
                        100,
                    )}%`
                  : "—"
              }
            />

            <InfoRow
              label="Clipping"
              value={
                result?.quality
                  ?.clipping_ratio !==
                undefined
                  ? `${(
                      result.quality.clipping_ratio *
                      100
                    ).toFixed(2)}%`
                  : "—"
              }
            />

          </div>

          {/* TEMPORAL */}

          <div className="card">

            <div className="cardTitle">
              TEMPORAL ASSESSMENT
            </div>

            <InfoRow
              label="Windows observed"
              value={
                result?.temporal
                  ?.windows_seen ??
                0
              }
            />

            <InfoRow
              label="Current risk"
              value={Math.round(
                result?.temporal
                  ?.current_risk ??
                  risk,
              )}
            />

            <InfoRow
              label="Maximum risk"
              value={Math.round(
                result?.temporal
                  ?.max_risk ??
                  risk,
              )}
            />

            <InfoRow
              label="Trend"
              value={
                result?.temporal?.trend ??
                "—"
              }
            />

            <InfoRow
              label="Persistence"
              value={
                result?.temporal
                  ?.persistence !==
                undefined
                  ? result.temporal.persistence.toFixed(
                      2,
                    )
                  : "—"
              }
            />

          </div>

          {/* MODEL RUNTIME */}

          <div className="card">

            <div className="cardTitle">
              MODEL RUNTIME
            </div>

            <InfoRow
              label="Model version"
              value={modelVersion}
            />

            <InfoRow
              label="Inference device"
              value={modelDevice}
            />

            <InfoRow
              label="Sample rate"
              value="16 kHz"
            />

            <InfoRow
              label="Window"
              value="10 seconds"
            />

            <InfoRow
              label="Last sequence"
              value={
                lastSequence ?? "—"
              }
            />

          </div>

          {/* STREAM TELEMETRY */}

          <div className="card">

            <div className="cardTitle">
              STREAM TELEMETRY
            </div>

            <InfoRow
              label="Session"
              value={
                sessionId
                  ? sessionId.slice(
                      0,
                      16,
                    ) + "..."
                  : "Not started"
              }
            />

            <InfoRow
              label="Chunks sent"
              value={chunksSent}
            />

            <InfoRow
              label="Stream"
              value={
                running
                  ? "CONNECTED"
                  : "STOPPED"
              }
            />

            <InfoRow
              label="Transport"
              value="REST streaming"
            />

          </div>

        </section>

        {/* SECURITY RESPONSE */}

        <section className="actionCard">

          <div>
            <div className="cardTitle">
              SECURITY RESPONSE
            </div>

            <h2>
              {level === "CRITICAL"
                ? "Callback / Escalation Required"
                : level === "HIGH"
                  ? "Secondary Verification Required"
                  : level === "MEDIUM"
                    ? "Verify Caller"
                    : level === "LOW"
                      ? "Continue"
                      : "Awaiting Voice Analysis"}
            </h2>

            <p>
  {level === "CRITICAL"
    ? "High-confidence impersonation risk detected. Callback or escalation is recommended."
    : level === "HIGH"
      ? "Significant voice-integrity risk detected. Secondary verification is recommended."
      : level === "MEDIUM"
        ? "Moderate voice-integrity risk detected. Verify the caller before proceeding."
        : level === "LOW"
          ? "Voice-integrity risk is currently low. Continue monitoring."
          : "Security response updates from the VaaniX risk engine."}
</p>
          </div>

          <div className="actions">

            <button
              onClick={() =>
                chooseAction("VERIFY")
              }
            >
              VERIFY
            </button>

            <button
              onClick={() =>
                chooseAction("MFA")
              }
            >
              MFA
            </button>

            <button
              onClick={() =>
                chooseAction("CALLBACK")
              }
            >
              CALLBACK
            </button>

            <button
              onClick={() =>
                chooseAction("ESCALATE")
              }
            >
              ESCALATE
            </button>

          </div>

        </section>

        {/* ACTIVITY */}

        <section className="activityCard">

          <div className="sectionHeader">

            <div>
              <span className="sectionTag">
                OBSERVABILITY
              </span>

              <h2>
                Live System Activity
              </h2>
            </div>

            <span className="observed">
              Backend events observed
            </span>

          </div>

          <div className="activityList">

            {activities.length === 0 ? (
              <div className="emptyActivity">
                Start a session to observe
                VaaniX processing activity.
              </div>
            ) : (
              activities.map(
                (activity, index) => (
                  <div
                    className="activity"
                    key={`${activity.time}-${index}`}
                  >
                    <span
                      className={`activityDot ${activity.type}`}
                    />

                    <span className="activityTime">
                      {activity.time}
                    </span>

                    <span>
                      {activity.text}
                    </span>
                  </div>
                ),
              )
            )}

          </div>

        </section>

        {/* INTEGRATION */}

        <section className="integration">

          <div>
            <strong>
              VAANIX INTEGRATION
            </strong>

            <span>
              REST API
            </span>

            <span>
              SESSION STREAMING
            </span>

            <span>
              LIVE ML RESULTS
            </span>
          </div>

          <div className="session">
            Session:{" "}
            {sessionId || "Not started"}
          </div>

        </section>

      </main>

      <footer>
        VaaniX • Voice Integrity Security •
        UpperSix
      </footer>

    </div>
  );
}

function PipelineStep({
  number,
  title,
  description,
  status,
  active,
}: {
  number: string;
  title: string;
  description: string;
  status: string;
  active: boolean;
}) {
  return (
    <div
      className={`pipelineStep ${
        active ? "active" : ""
      }`}
    >
      <div className="pipelineNumber">
        {number}
      </div>

      <div className="pipelineTitle">
        {title}
      </div>

      <div className="pipelineDescription">
        {description}
      </div>

      <div className="pipelineStatus">
        {status}
      </div>
    </div>
  );
}
function Arrow() {
  return (
    <div className="pipelineArrow">
      →
    </div>
  );
}
function Evidence({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const percentage = Math.max(
    0,
    Math.min(100, value * 100),
  );

  return (
    <div className="evidence">

      <div>
        <span>
          {label}
        </span>

        <strong>
          {Math.round(
            percentage,
          )}
          %
        </strong>
      </div>

      <div className="bar">
        <div
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>

    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="infoRow">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}

export default App;
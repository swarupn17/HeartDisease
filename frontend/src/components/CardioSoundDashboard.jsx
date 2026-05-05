import React, { useMemo, useState } from 'react';

const DEFAULT_CONFIDENCE = [
  { label: 'Normal', value: 72 },
  { label: 'Murmur', value: 12 },
  { label: 'Extrastole', value: 9 },
  { label: 'Artifact', value: 7 },
];

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '0%';
  }

  return `${Math.round(value)}%`;
}

function AlertIcon({ tone }) {
  if (tone === 'normal') {
    return (
      <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-emerald-300" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M20 6L9 17l-5-5" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-rose-300" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3l-8.47-14.14a2 2 0 0 0-3.42 0Z" />
    </svg>
  );
}

function ProgressRow({ label, value, accentClass }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-300">
        <span>{label}</span>
        <span className="tabular-nums text-slate-100">{formatPercent(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800/90 ring-1 ring-white/5">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${accentClass} shadow-[0_0_18px_rgba(34,211,238,0.35)] transition-all duration-500`}
          style={{ width: `${Math.max(0, Math.min(100, value || 0))}%` }}
        />
      </div>
    </div>
  );
}

export default function CardioSoundDashboard({
  initialPrediction = 'Abnormal',
  initialConfidence = 94,
  initialConfidenceBreakdown = DEFAULT_CONFIDENCE,
  initialGradcamImageUrl = '',
  initialExplanation = 'The model is awaiting analysis output. Upload a heart sound recording to generate a prediction, explainability map, and clinical summary.',
  isLoading = false,
  error = '',
  onAnalyze,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const normalizedPrediction = (initialPrediction || 'Abnormal').toLowerCase();
  const isNormal = normalizedPrediction === 'normal';
  const tone = isNormal ? 'normal' : 'abnormal';
  const confidenceBreakdown = useMemo(() => {
    if (Array.isArray(initialConfidenceBreakdown) && initialConfidenceBreakdown.length > 0) {
      return initialConfidenceBreakdown;
    }

    return DEFAULT_CONFIDENCE;
  }, [initialConfidenceBreakdown]);

  const handleFile = (file) => {
    if (!file) {
      return;
    }

    setSelectedFile(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];
    handleFile(file);
  };

  const handleAnalyze = () => {
    if (typeof onAnalyze === 'function' && selectedFile) {
      onAnalyze(selectedFile);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute left-[-8rem] top-[-8rem] h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute right-[-6rem] top-24 h-80 w-80 rounded-full bg-violet-500/10 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(148,163,184,0.08),transparent_45%)]" />
      </div>

      <div className="mx-auto flex min-h-screen w-full max-w-7xl items-center px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid w-full gap-6 lg:grid-cols-[0.92fr_1.08fr]">
          <section className="rounded-3xl border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur-xl lg:p-8">
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300/80">CardioSound</p>
                <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Upload & Analyze</h1>
                <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                  Drop a `.wav` heart sound recording and generate a prediction, confidence breakdown, and Grad-CAM explanation.
                </p>
              </div>
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-3 py-2 text-right">
                <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-200/70">Status</p>
                <p className="text-sm font-medium text-cyan-100">Ready</p>
              </div>
            </div>

            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`group relative flex min-h-[280px] flex-col items-center justify-center rounded-3xl border border-dashed px-6 py-8 text-center transition-all duration-300 ${
                isDragging
                  ? 'border-cyan-300 bg-cyan-400/10 shadow-[0_0_0_1px_rgba(34,211,238,0.25),0_0_45px_rgba(34,211,238,0.12)]'
                  : 'border-slate-700 bg-slate-950/50'
              }`}
            >
              <div className="pointer-events-none absolute inset-0 rounded-3xl bg-[linear-gradient(135deg,rgba(34,211,238,0.06),transparent_35%,rgba(168,85,247,0.08))] opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              <div className="relative z-10 flex max-w-sm flex-col items-center gap-4">
                <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-4 shadow-[0_0_32px_rgba(34,211,238,0.12)]">
                  <svg viewBox="0 0 24 24" className="h-8 w-8 text-cyan-200" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 16V4" />
                    <path d="M8 8l4-4 4 4" />
                    <path d="M4 20h16" />
                  </svg>
                </div>
                <div>
                  <p className="text-lg font-medium text-white">Drop your audio file here</p>
                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    Drag and drop a `.wav` file or choose one manually.
                  </p>
                </div>
                <label className="inline-flex cursor-pointer items-center justify-center rounded-full border border-cyan-400/30 bg-cyan-400/10 px-5 py-3 text-sm font-medium text-cyan-100 transition-all duration-300 hover:-translate-y-0.5 hover:border-cyan-300 hover:bg-cyan-400/20 hover:shadow-[0_0_24px_rgba(34,211,238,0.22)]">
                  <input
                    type="file"
                    accept=".wav,audio/wav"
                    className="hidden"
                    onChange={(event) => handleFile(event.target.files?.[0])}
                  />
                  Choose `.wav` file
                </label>
                <p className="text-xs text-slate-500">Supported input: cardiac phonocardiogram recordings</p>
              </div>
            </div>

            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.28em] text-slate-500">Selected File</p>
                <p className="truncate text-sm text-slate-200">{selectedFile ? selectedFile.name : 'No file selected'}</p>
              </div>

              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!selectedFile || isLoading}
                className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-cyan-400 to-violet-500 px-6 py-3 text-sm font-semibold text-slate-950 transition-all duration-300 hover:scale-[1.01] hover:shadow-[0_0_28px_rgba(56,189,248,0.35)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isLoading ? 'Analyzing...' : 'Analyze Recording'}
              </button>
            </div>
          </section>

          <section className="rounded-3xl border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-violet-950/20 backdrop-blur-xl lg:p-8">
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.28em] text-violet-300/80">Diagnostic Results</p>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white">Clinical Overview</h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
                  The prediction card, class confidence, and explainability output are laid out for a direct clinician review flow.
                </p>
              </div>
              <div className={`rounded-2xl border px-3 py-2 ${isNormal ? 'border-emerald-400/20 bg-emerald-400/10' : 'border-rose-400/20 bg-rose-400/10'}`}>
                <p className="text-[11px] uppercase tracking-[0.3em] text-slate-300/70">Prediction</p>
                <p className={`text-sm font-medium ${isNormal ? 'text-emerald-200' : 'text-rose-200'}`}>{initialPrediction}</p>
              </div>
            </div>

            {error ? (
              <div className="mb-5 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {error}
              </div>
            ) : null}

            <div className={`mb-6 rounded-3xl border p-5 ${isNormal ? 'border-emerald-400/20 bg-emerald-500/10' : 'border-rose-400/20 bg-rose-500/10'} shadow-[0_0_32px_rgba(15,23,42,0.3)]`}>
              <div className="flex items-start gap-3">
                <div className={`mt-0.5 rounded-full p-2 ${isNormal ? 'bg-emerald-400/10' : 'bg-rose-400/10'}`}>
                  <AlertIcon tone={tone} />
                </div>
                <div className="min-w-0">
                  <p className={`text-sm font-semibold uppercase tracking-[0.25em] ${isNormal ? 'text-emerald-200' : 'text-rose-200'}`}>
                    {isNormal ? 'Normal' : 'Abnormal'}
                  </p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">
                    {isNormal ? 'Low concern signal detected' : 'Abnormal rhythm signature detected'}
                  </h3>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
                    Confidence score: <span className="font-semibold text-white">{formatPercent(initialConfidence)}</span>. This panel is designed to update directly from the API response.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1fr_0.92fr]">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-white">Confidence Breakdown</h3>
                    <p className="mt-1 text-sm text-slate-400">Probability distribution across the four output classes.</p>
                  </div>
                </div>
                <div className="space-y-5">
                  {confidenceBreakdown.map((item, index) => {
                    const accents = [
                      'from-cyan-300 to-cyan-500',
                      'from-violet-300 to-violet-500',
                      'from-fuchsia-300 to-fuchsia-500',
                      'from-amber-300 to-amber-500',
                    ];

                    return (
                      <ProgressRow
                        key={item.label}
                        label={item.label}
                        value={typeof item.value === 'number' ? item.value : 0}
                        accentClass={accents[index % accents.length]}
                      />
                    );
                  })}
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-semibold text-white">Explainability Viewer</h3>
                    <p className="mt-1 text-sm text-slate-400">Grad-CAM heatmap and generated explanation.</p>
                  </div>
                  {isLoading ? (
                    <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-100">
                      Loading
                    </span>
                  ) : null}
                </div>

                <div className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-950/60">
                  {initialGradcamImageUrl ? (
                    <img
                      src={initialGradcamImageUrl}
                      alt="Grad-CAM explanation"
                      className="h-72 w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-72 items-center justify-center px-6 text-center text-sm leading-6 text-slate-500">
                      The Grad-CAM image will appear here after analysis.
                    </div>
                  )}
                </div>

                <div className="mt-4 border-l-2 border-cyan-400/60 bg-white/5 px-4 py-3">
                  <p className="text-xs uppercase tracking-[0.28em] text-cyan-200/70">LLM Explanation</p>
                  <p className="mt-2 text-sm italic leading-6 text-slate-300">
                    {initialExplanation}
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

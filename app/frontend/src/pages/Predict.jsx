import React, { useState } from "react";
import { analysisData } from "../data/analysisData";
import {
  fieldGroups, numericMeta, categoricalMeta, ALL_NUMERIC, ALL_CATEGORICAL,
  addDerivedFeatures, estimateEnergy, estimateRisk, estimateCluster,
} from "../data/featureMeta";

export default function Predict() {
  const { app } = analysisData;
  const meta = app.metadata;

  // Initialise defaults from the metadata JSON
  const initial = {};
  for (const f of ALL_CATEGORICAL) {
    initial[f] = categoricalMeta[f].default || categoricalMeta[f].options[0];
  }
  for (const f of ALL_NUMERIC) {
    initial[f] = numericMeta[f].median;
  }

  const [values, setValues] = useState(initial);
  const [result, setResult] = useState(null);

  const handleNum = (f, v) => setValues({ ...values, [f]: v });
  const handleCat = (f, v) => setValues({ ...values, [f]: v });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const derived = addDerivedFeatures(values);

    try {
      const baseUrl = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${baseUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(derived),
      });
      if (response.ok) {
        const data = await response.json();
        const cluster = estimateCluster(derived);
        const colors = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };
        const score = Math.round((data.probabilities[data.risk] || 0.5) * 100);
        setResult({
          energy: data.energy_kwh,
          risk: {
            risk: data.risk,
            score,
            color: colors[data.risk] || "#22c55e",
            confidence: data.probabilities,
          },
          cluster,
          derived,
          isLiveModel: true,
        });
        return;
      }
    } catch (err) {
      // Fallback to client heuristics when backend service is unreachable
    }

    const energy = estimateEnergy(derived);
    const risk = estimateRisk(derived);
    const cluster = estimateCluster(derived);
    setResult({ energy, risk, cluster, derived, isLiveModel: false });
  };

  const reset = () => {
    const fresh = {};
    for (const f of ALL_CATEGORICAL) fresh[f] = categoricalMeta[f].default || categoricalMeta[f].options[0];
    for (const f of ALL_NUMERIC) fresh[f] = numericMeta[f].median;
    setValues(fresh); setResult(null);
  };

  return (
    <div className="space-y-8 pb-12">
      <div className="text-center py-8">
        <h1 className="text-3xl font-bold text-white mb-2">Maintenance Risk &amp; Energy Predictor</h1>
        <p className="text-slate-300">Enter machine readings to estimate maintenance risk and energy consumption</p>
      </div>

      <div className="card">
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="p-3 rounded-lg bg-surface-2/30 border border-slate-700/40">
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
              Track 2 • Classifier
            </span>
            <p className="text-sm font-semibold text-white mt-1">{meta.classifier.name}</p>
            <p className="text-xs text-slate-400">Target: maintenance_risk (Low/Med/High)</p>
            <p className="text-xs text-cyan-300 mt-1">
              Accuracy {meta.classifier.accuracy.toFixed(2)} &middot; F1 {meta.classifier.weightedF1.toFixed(3)}
            </p>
          </div>
          <div className="p-3 rounded-lg bg-surface-2/30 border border-slate-700/40">
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
              Track 1 • Regressor
            </span>
            <p className="text-sm font-semibold text-white mt-1">{meta.regressor.name}</p>
            <p className="text-xs text-slate-400">Target: machine_energy_kwh</p>
            <p className="text-xs text-cyan-300 mt-1">
              R² {meta.regressor.r2.toFixed(3)} &middot; MAE {meta.regressor.mae.toFixed(2)} kWh
            </p>
          </div>
          <div className="p-3 rounded-lg bg-surface-2/30 border border-slate-700/40">
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
              Track 3 • Clustering
            </span>
            <p className="text-sm font-semibold text-white mt-1">K-Means &amp; Agglomerative (Ward)</p>
            <p className="text-xs text-slate-400">Target: Unsupervised (k=5 Profiles)</p>
            <p className="text-xs text-cyan-300 mt-1">Selected k=5 &middot; Davies-Bouldin 3.02</p>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="sm:hidden mb-4">
            <label className="text-sm text-slate-300">Pick a section</label>
            <SectionSelect groups={Object.keys(fieldGroups)} />
          </div>

          <div className="hidden sm:flex gap-2 mb-4 overflow-x-auto">
            {Object.keys(fieldGroups).map((g) => (
              <TabButton key={g} label={g} active={true} />
            ))}
          </div>

          <div className="space-y-6">
            {Object.entries(fieldGroups).map(([group, fields]) => (
              <FormFieldGroup
                key={group}
                groupName={group}
                fields={fields.filter((f) => [...ALL_CATEGORICAL, ...ALL_NUMERIC].includes(f))}
                values={values}
                onNum={handleNum}
                onCat={handleCat}
              />
            ))}
          </div>

          <div className="mt-6 flex gap-3">
            <button type="submit" className="btn-primary">
              Predict
            </button>
            <button type="button" onClick={reset} className="btn-secondary">
              Reset to defaults
            </button>
          </div>
        </form>
      </div>

      {result && <ResultCard result={result} values={values} />}
    </div>
  );
}

function SectionSelect({ groups }) {
  return (
    <select className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-slate-600 text-sm text-white">
      {groups.map((g) => <option key={g}>{g}</option>)}
    </select>
  );
}
function TabButton({ label, active }) {
  return (
    <button type="button" className={`px-4 py-2 rounded-md text-sm font-medium ${
      active ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40"
        : "text-slate-400 hover:text-cyan-300"
    }`}>{label}</button>
  );
}

function FormFieldGroup({ groupName, fields, values, onNum, onCat }) {
  return (
    <div>
      <h4 className="text-sm font-semibold text-slate-300 mb-3">{groupName}</h4>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {fields.map((f) => {
          const isCat = f in categoricalMeta;
          const pretty = f.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
          if (isCat) {
            const meta = categoricalMeta[f];
            return (
              <div key={f}>
                <label className="text-xs font-medium text-slate-300">{pretty}</label>
                <select
                  value={values[f]}
                  onChange={(e) => onCat(f, e.target.value)}
                  className="mt-1 w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 cursor-pointer"
                >
                  {meta.options.map((opt) => <option key={opt} value={opt} className="bg-slate-900 text-white">{opt.trim()}</option>)}
                </select>
              </div>
            );
          }
          const meta = numericMeta[f];
          return (
            <div key={f}>
              <label className="text-xs font-medium text-slate-300">{pretty}</label>
              <input
                type="number"
                value={values[f]}
                onChange={(e) => onNum(f, parseFloat(e.target.value) || 0)}
                min={meta.min} max={meta.max} step="any"
                className="mt-1 w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              />
              <div className="text-[10px] text-slate-400 mt-0.5">
                range: {meta.min.toFixed(1)} – {meta.max.toFixed(1)} | median: {meta.median}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ResultCard({ result, values }) {
  const { energy, risk, cluster, isLiveModel } = result;
  const { risk: riskLabel, score, color } = risk;
  return (
    <div className="card border border-cyan-500/30">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h3 className="text-cyan-300 font-medium">Model Inference Results (All 3 Project Tracks)</h3>
        {isLiveModel ? (
          <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Live Pickle Models (.pkl) Active
          </span>
        ) : (
          <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-medium">
            Fallback Heuristic Mode
          </span>
        )}
      </div>
      <div className="grid md:grid-cols-3 gap-6">
        {/* Track 2: Classifier */}
        <div className="text-center p-4 rounded-lg bg-surface-2/40 border border-slate-600/20">
          <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
            Track 2 • Classifier
          </span>
          <div className="text-3xl font-bold mt-2 mb-1" style={{ color }}>
            {riskLabel}
          </div>
          <p className="text-sm font-medium text-slate-200">Maintenance Risk Level</p>
          <div className="mt-2 w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all" style={{ width: `${score}%`, backgroundColor: color }} />
          </div>
          <p className="text-xs text-slate-400 mt-1">Risk score: {score}/100</p>
          <div className="mt-3 grid grid-cols-3 gap-1 text-[11px] text-slate-300">
            <div>Low: {Math.round(risk.confidence.Low * 100)}%</div>
            <div>Med: {Math.round(risk.confidence.Medium * 100)}%</div>
            <div>High: {Math.round(risk.confidence.High * 100)}%</div>
          </div>
        </div>

        {/* Track 1: Regressor */}
        <div className="text-center p-4 rounded-lg bg-surface-2/40 border border-slate-600/20">
          <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
            Track 1 • Regressor
          </span>
          <div className="text-3xl font-bold text-cyan-300 mt-2 mb-1">{energy.toFixed(2)} <span className="text-sm font-normal text-slate-400">kWh</span></div>
          <p className="text-sm font-medium text-slate-200">Predicted Energy Draw</p>
          <p className="text-xs text-slate-400 mt-3">±{analysisData.app.metadata.regressor.mae.toFixed(2)} kWh (test MAE)</p>
        </div>

        {/* Track 3: Clustering */}
        <div className="text-center p-4 rounded-lg bg-surface-2/40 border border-slate-600/20">
          <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
            Track 3 • Clustering (k=5)
          </span>
          <div className="text-xl font-bold mt-2 mb-1" style={{ color: cluster.color }}>{cluster.name}</div>
          <p className="text-sm font-medium text-slate-200">{cluster.tag}</p>
          <p className="text-xs text-slate-400 mt-2 leading-relaxed">{cluster.desc}</p>
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-slate-700/30">
        <p className="text-xs text-slate-400">
          {isLiveModel ? (
            <span className="text-emerald-400 font-medium">
              ✓ Prediction evaluated live from scikit-learn models (final_classifier.pkl &amp; final_regressor.pkl) via FastAPI.
            </span>
          ) : (
            <span>
              Note: Currently running in client-side fallback mode. To evaluate through the real scikit-learn pickle models, start the FastAPI service in <code>app/main.py</code> or configure <code>VITE_API_URL</code>.
            </span>
          )}
        </p>
      </div>
    </div>
  );
}

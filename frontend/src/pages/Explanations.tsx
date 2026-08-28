import React, { useState } from 'react';
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  Clock,
  Compass,
  Cpu,
  HelpCircle,
  Layers,
  Orbit,
  Scale,
  ShieldCheck,
  TrendingDown,
  Zap,
} from 'lucide-react';
import { DisclaimerBanner } from '../components/DisclaimerBanner';

export const Explanations: React.FC = () => {
  const [openSection, setOpenSection] = useState<string | null>('conjunction');

  const toggleSection = (key: string) => {
    setOpenSection(openSection === key ? null : key);
  };

  const sections = [
    {
      id: 'conjunction',
      title: '1. What is an Orbital Conjunction & CDM?',
      icon: Orbit,
      summary: 'A close spatial approach between two orbiting satellites or debris objects.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            An <strong>Orbital Conjunction</strong> occurs when the predicted trajectories of two resident space objects (e.g., an operational spacecraft and a piece of orbital debris) bring them into close geometric proximity in Earth orbit.
          </p>
          <p>
            Space surveillance networks (such as ESA or US Space Command) generate standardized <strong>Conjunction Data Messages (CDMs)</strong>. Each CDM contains:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-400">
            <li><strong>Time of Closest Approach (TCA):</strong> The exact timestamp when geometric separation is minimized.</li>
            <li><strong>Relative Geometry:</strong> Miss distance (m) and relative velocity vector (m/s).</li>
            <li><strong>Covariance Matrices:</strong> Position and velocity error dispersions for primary and secondary objects.</li>
            <li><strong>Space Weather & Environmental Indices:</strong> Atmospheric solar radio flux (F10.7) and geomagnetic index (AP).</li>
          </ul>
        </div>
      ),
    },
    {
      id: 'log10_risk',
      title: '2. What is Log10 Collision Risk (log10 Pc)?',
      icon: Compass,
      summary: 'Standard logarithmic transformation of collision probability to handle orders of magnitude.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            The collision probability Pc is calculated by integrating probability density functions over hard-body collision spheres. Because Pc spans dozens of orders of magnitude (from 10^-30 to 10^-2), direct linear regression produces extreme numerical instability.
          </p>
          <p>
            ORVEXA models the target as the continuous <strong>log10(Pc)</strong> risk value:
          </p>
          <div className="p-3 bg-space-950 rounded border border-slate-800 font-mono text-cyan-300 text-center">
            y = log10(Pc) ∈ [-30.0, 0.0]
          </div>
          <ul className="list-disc pl-5 space-y-1 text-slate-400">
            <li><strong>log10(Pc) &ge; -4.0:</strong> Critical Risk (Pc &ge; 10^-4), operational threshold for avoidance maneuver planning.</li>
            <li><strong>-6.0 &le; log10(Pc) &lt; -4.0:</strong> Moderate Risk (10^-6 &le; Pc &lt; 10^-4), heightened tracking required.</li>
            <li><strong>log10(Pc) &lt; -15.0:</strong> Negligible Risk (Pc &le; 10^-15), nominal pass.</li>
          </ul>
        </div>
      ),
    },
    {
      id: 'tcn',
      title: '3. What is a Causal Temporal Convolutional Network (TCN)?',
      icon: Cpu,
      summary: '1D dilated convolutions with strict causal masking to ingest sequential approach observations without future leakage.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            As a conjunction approaches TCA, multiple CDMs are received sequentially. Recurrent neural networks (RNN/LSTM) suffer from gradient vanishing and sequential bottlenecks, while standard transformers lack strict inductive bias for temporal causal receptive fields.
          </p>
          <p>
            The ORVEXA <strong>Causal TCN</strong> utilizes:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-400">
            <li><strong>Causal 1D Dilated Convolutions:</strong> Output at timestep t depends strictly on timesteps &le; t, guaranteeing zero future leakage.</li>
            <li><strong>Exponential Dilation Factors (d = 1, 2, 4):</strong> Expands receptive field across entire 23-CDM observation sequences without adding excessive parameters.</li>
            <li><strong>Left-Zero Padding:</strong> Standardizes variable length CDM sequences to a uniform tensor without shifting temporal alignment.</li>
          </ul>
        </div>
      ),
    },
    {
      id: 'quantile',
      title: '4. What is Multi-Quantile Regression & Monotonic Step Head?',
      icon: Layers,
      summary: 'Pinball loss optimization for quantiles with softplus parameterization to prevent quantile crossing.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Instead of predicting a single mean scalar, ORVEXA estimates a full predictive distribution across 7 quantiles: &tau; &isin; [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95].
          </p>
          <p>
            Trained with the <strong>Multi-Quantile Pinball Loss (Asymmetric Linear Loss)</strong>:
          </p>
          <div className="p-3 bg-space-950 rounded border border-slate-800 font-mono text-cyan-300 text-center">
            L_&tau;(y, y_hat) = max( &tau;(y - y_hat), (1 - &tau;)(y_hat - y) )
          </div>
          <p>
            To strictly prevent <strong>Quantile Crossing</strong> (where a higher quantile would predict a lower value), ORVEXA enforces an incremental softplus parameterization:
          </p>
          <div className="p-3 bg-space-950 rounded border border-slate-800 font-mono text-amber-300 text-center">
            q_0 = W_0 h + b_0, &nbsp;&nbsp; q_k = q_(k-1) + Softplus(W_k h + b_k) + &epsilon;
          </div>
          <p className="text-slate-400">
            This mathematically guarantees q_0.05 &le; q_0.10 &le; ... &le; q_0.95 with zero empirical violations.
          </p>
        </div>
      ),
    },
    {
      id: 'cqr',
      title: '5. What is Conformalized Quantile Regression (CQR)?',
      icon: Scale,
      summary: 'Distribution-free, finite-sample calibrated prediction intervals with theoretical coverage guarantees.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Raw quantile neural networks can suffer from miscalibration or undercoverage on out-of-distribution inputs. <strong>Conformalized Quantile Regression (CQR)</strong> wraps the trained quantile model with a calibration layer.
          </p>
          <p>
            On a strictly held-out Calibration partition (n_cal &approx; 1,000 events), nonconformity scores are computed:
          </p>
          <div className="p-3 bg-space-950 rounded border border-slate-800 font-mono text-cyan-300 text-center">
            s_i = max( q_0.05(X_i) - Y_i, Y_i - q_0.95(X_i) )
          </div>
          <p>
            The critical shift q_hat is selected as the (1 - &alpha;)-th empirical quantile of calibration nonconformity scores. The final prediction interval is:
          </p>
          <div className="p-3 bg-space-950 rounded border border-slate-800 font-mono text-emerald-300 text-center">
            C(X) = [ q_0.05(X) - q_hat, q_0.95(X) + q_hat ]
          </div>
          <p className="text-slate-400">
            <strong>Key Benefit:</strong> Provides guaranteed &ge; 90% coverage while reducing interval widths by ~49% compared to constant-width residual conformal intervals.
          </p>
        </div>
      ),
    },
    {
      id: 'degradation',
      title: '6. Why Does Performance Degrade with Longer Warning Horizons?',
      icon: TrendingDown,
      summary: 'Atmospheric density fluctuations and non-linear astrodynamic dispersion over multi-day propagation.',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Predicting collision risk 6 days in advance (H6 = 144h) is physically distinct from predicting it 2 days in advance (H2 = 48h):
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-400">
            <li><strong>Atmospheric Drag Uncertainty:</strong> Space weather fluctuations (solar flares, geomagnetic storms) cause unmodeled density variations in Low Earth Orbit (LEO).</li>
            <li><strong>Covariance Growth:</strong> Small initial tracking velocity errors (&plusmn; 1 mm/s) expand into kilometers of positional uncertainty over hundreds of orbits.</li>
            <li><strong>Sparse Early Tracking:</strong> At 6 days before TCA, radar tracking passes are sparse (often only 1 or 2 CDMs).</li>
          </ul>
          <p className="text-amber-300 font-semibold">
            Result: Point prediction accuracy drops at long horizons (H6 point R² = -0.166). This is why probabilistic uncertainty bounds via CQR are essential for decision support.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-cyan-400" />
            <span>Scientific Knowledge Base & Technical Glossary</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Core mathematical and astrodynamical concepts underlying the ORVEXA conjunction risk framework.
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 bg-space-850 rounded-lg border border-slate-800 text-slate-300">
          Academic Research Reference
        </div>
      </div>

      {/* Accordion List */}
      <div className="space-y-3">
        {sections.map((sec) => {
          const isOpen = openSection === sec.id;
          const Icon = sec.icon;
          return (
            <div
              key={sec.id}
              className="panel-card overflow-hidden transition-all border border-slate-800/90"
            >
              <button
                onClick={() => toggleSection(sec.id)}
                className="w-full p-4 text-left flex items-center justify-between hover:bg-space-850/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${isOpen ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' : 'bg-space-850 text-slate-400 border border-slate-800'}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className={`text-sm font-semibold ${isOpen ? 'text-cyan-300' : 'text-white'}`}>
                      {sec.title}
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">{sec.summary}</p>
                  </div>
                </div>

                <div className="text-slate-400">
                  {isOpen ? <ChevronUp className="w-4 h-4 text-cyan-400" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {isOpen && (
                <div className="p-5 border-t border-slate-800/80 bg-space-950/40 animate-fadeIn">
                  {sec.content}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

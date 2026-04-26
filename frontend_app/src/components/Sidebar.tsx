"use client";
import { useState } from 'react';
import { IncidentState, ModelConfig } from '@/hooks/useEnvironment';

const TASKS = [
  { value: 'single_service_failure', label: 'Easy: Single Failure', emoji: '🟢' },
  { value: 'cascading_failure', label: 'Medium: Cascading', emoji: '🟡' },
  { value: 'hidden_root_cause', label: 'Hard: Hidden Root', emoji: '🟠' },
  { value: 'chaos_cascade', label: 'Hard: Chaos Cascade', emoji: '🔴' },
  { value: 'multi_root_cause', label: 'Expert: Multi-Root', emoji: '💀' },
  { value: 'random_incident', label: 'Random Incident', emoji: '🎲' },
];

interface SidebarProps {
  collapsed: boolean;
  state?: IncidentState | null;
  onReset?: (taskName: string, chaosMode: boolean) => void;
  selectedTask?: string;
  onSelectedTaskChange?: (taskName: string) => void;
  chaosMode?: boolean;
  onChaosModeChange?: (enabled: boolean) => void;
  modelConfig?: ModelConfig;
  onModelConfigChange?: (cfg: ModelConfig) => void;
  modelInfo?: any | null;
  isAutoPilotRunning?: boolean;
  onStartAutoPilot?: () => void;
  onStopAutoPilot?: () => void;
  liveScore?: { score: number; breakdown: Record<string, number> } | null;
}

export default function Sidebar({
  collapsed, state, onReset,
  selectedTask, onSelectedTaskChange,
  chaosMode, onChaosModeChange,
  modelConfig, onModelConfigChange,
  modelInfo,
  isAutoPilotRunning, onStartAutoPilot, onStopAutoPilot,
  liveScore,
}: SidebarProps) {
  const localSelectedTask = selectedTask ?? 'random_incident';
  const localChaosMode = chaosMode ?? true;
  const localModel = modelConfig ?? {
    base_model: 'sft_merged_1p5b_v5',
    adapter_path: 'trained_model_1p5b_v5',
    device: 'auto' as const,
  };

  const handleReset = () => {
    if (onReset) onReset(localSelectedTask, localChaosMode);
  };

  return (
    <nav
      className={`bg-surface-container-lowest text-on-surface-variant font-manrope text-[11px] uppercase tracking-widest font-semibold h-screen fixed left-0 top-0 border-r border-outline-variant/10 flex flex-col pt-16 pb-6 z-40 sidebar-transition ${
        collapsed ? 'collapsed-sidebar' : 'w-60'
      }`}
      id="sidebar"
    >
      <div className="px-6 mb-8 whitespace-nowrap overflow-hidden">
        <h1 className="text-sm font-bold text-on-surface font-h2 mb-0.5 tracking-tight">CONTROL CENTER</h1>
        <p className="text-[10px] text-outline normal-case tracking-normal font-medium opacity-70 truncate">
          {state?.episode_id ? `Session: ${state.episode_id.split('-')[0]}` : 'Waiting for connection...'}
        </p>
      </div>
      
      <div className="flex flex-col gap-0.5 w-full flex-grow px-3 overflow-hidden">
        <a className="flex items-center gap-3 px-3 py-2 bg-primary/5 text-primary border-r-2 border-primary rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">hub</span>
          <span className="text-[11px]">Service Map</span>
        </a>
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">segment</span>
          <span className="text-[11px]">Activity Log</span>
        </a>
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">monitoring</span>
          <span className="text-[11px]">Performance</span>
        </a>

        {/* Divider */}
        <div className="border-t border-outline-variant/10 my-3" />

        {/* Task Selector */}
        <div className="px-2 mb-2">
          <label className="text-[9px] text-outline/70 normal-case tracking-normal block mb-1.5">Scenario</label>
          <select
            id="task-selector"
            value={localSelectedTask}
            onChange={(e) => onSelectedTaskChange?.(e.target.value)}
            className="w-full bg-surface-container text-on-surface text-[11px] normal-case tracking-normal font-medium py-1.5 px-2 rounded border border-outline-variant/20 focus:border-primary focus:outline-none transition-colors"
          >
            {TASKS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.emoji} {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Chaos Mode Toggle */}
        <div className="px-2 mb-2 flex items-center justify-between">
          <label className="text-[9px] text-outline/70 normal-case tracking-normal">Chaos Mode</label>
          <button
            id="chaos-toggle"
            onClick={() => onChaosModeChange?.(!localChaosMode)}
            className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${
              localChaosMode
                ? 'bg-error/60'
                : 'bg-outline-variant/20'
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform duration-200 ${
                localChaosMode
                  ? 'translate-x-5 bg-error'
                  : 'translate-x-0 bg-outline/50'
              }`}
            />
            {localChaosMode && (
              <span className="absolute -top-1 -right-1 text-[10px]">⚡</span>
            )}
          </button>
        </div>

        {/* Model Config */}
        <div className="px-2 mb-2">
          <label className="text-[9px] text-outline/70 normal-case tracking-normal block mb-1.5">Model (backend /predict)</label>
          <input
            value={localModel.base_model}
            onChange={(e) => onModelConfigChange?.({ ...localModel, base_model: e.target.value })}
            className="w-full bg-surface-container text-on-surface text-[11px] normal-case tracking-normal font-medium py-1.5 px-2 rounded border border-outline-variant/20 focus:border-primary focus:outline-none transition-colors mb-1.5"
            placeholder="Base model (HF id or local path)"
          />
          <input
            value={localModel.adapter_path}
            onChange={(e) => onModelConfigChange?.({ ...localModel, adapter_path: e.target.value })}
            className="w-full bg-surface-container text-on-surface text-[11px] normal-case tracking-normal font-medium py-1.5 px-2 rounded border border-outline-variant/20 focus:border-primary focus:outline-none transition-colors mb-1.5"
            placeholder="Adapter path (folder)"
          />
          <select
            value={localModel.device}
            onChange={(e) => onModelConfigChange?.({ ...localModel, device: e.target.value as any })}
            className="w-full bg-surface-container text-on-surface text-[11px] normal-case tracking-normal font-medium py-1.5 px-2 rounded border border-outline-variant/20 focus:border-primary focus:outline-none transition-colors"
          >
            <option value="auto">auto</option>
            <option value="mps">mps</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
          </select>
          {modelInfo && (
            <div className="mt-2 text-[10px] normal-case tracking-normal text-outline/70">
              Model loaded: <span className="text-on-surface">{String(modelInfo.loaded)}</span>
              {modelInfo.device && (
                <> · device: <span className="text-on-surface">{modelInfo.device}</span></>
              )}
              {modelInfo.error && (
                <div className="mt-1 text-error">Model error: {String(modelInfo.error)}</div>
              )}
            </div>
          )}
        </div>

        {/* Live Score Display */}
        {liveScore && (
          <div className="px-2 mb-2">
            <label className="text-[9px] text-outline/70 normal-case tracking-normal block mb-1">Live Score</label>
            <div className="bg-surface-container rounded p-2 border border-outline-variant/10">
              <div className="text-lg font-bold text-on-surface text-center normal-case tracking-normal mb-1">
                {liveScore.score.toFixed(3)}
              </div>
              <div className="space-y-0.5">
                {Object.entries(liveScore.breakdown).map(([key, val]) => {
                  const maxVals: Record<string, number> = { recovery: 0.35, efficiency: 0.20, diagnostics: 0.15, ordering: 0.20, memory: 0.10 };
                  const max = maxVals[key] || 0.2;
                  const pct = max > 0 ? (val / max) * 100 : 0;
                  return (
                    <div key={key} className="flex items-center gap-1.5">
                      <span className="text-[8px] text-outline/60 w-14 normal-case tracking-normal truncate">{key}</span>
                      <div className="flex-1 h-1.5 bg-outline-variant/10 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            pct >= 90 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${Math.min(pct, 100)}%` }}
                        />
                      </div>
                      <span className="text-[8px] text-outline/50 w-8 text-right normal-case">{val.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="px-4 mt-auto mb-2 overflow-hidden flex flex-col gap-2">
        {/* Auto-Pilot Button */}
        <button 
          id="auto-pilot-btn"
          onClick={isAutoPilotRunning ? onStopAutoPilot : onStartAutoPilot}
          className={`w-full text-[10px] font-semibold tracking-wider uppercase py-2.5 rounded transition-all duration-300 whitespace-nowrap flex justify-center items-center gap-2 ${
            isAutoPilotRunning
              ? 'bg-warning/20 border border-warning/40 text-warning animate-pulse'
              : 'bg-primary/10 border border-primary/20 text-primary hover:bg-primary/20'
          }`}
        >
          <span className="material-symbols-outlined text-sm">
            {isAutoPilotRunning ? 'stop_circle' : 'smart_toy'}
          </span>
          {isAutoPilotRunning ? 'Stop AI' : 'Auto-Pilot'}
        </button>

        {/* Reset Button */}
        <button 
          id="reset-btn"
          onClick={handleReset}
          className="w-full bg-error/10 border border-error/20 text-error text-[10px] font-semibold tracking-wider uppercase py-2 rounded transition-colors hover:bg-error/20 whitespace-nowrap flex justify-center items-center gap-2"
        >
          <span className="material-symbols-outlined text-sm">refresh</span>
          Restart Sim
        </button>
      </div>
    </nav>
  );
}

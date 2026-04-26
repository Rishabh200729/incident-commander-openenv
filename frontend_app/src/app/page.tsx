"use client";
import { useState, useEffect, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ServiceMap from "@/components/ServiceMap";
import ActivityLog from "@/components/ActivityLog";
import MetricsGauges from "@/components/MetricsGauges";
import PerformanceTrend from "@/components/PerformanceTrend";
import ReportModal from "@/components/ReportModal";
import { useEnvironment } from "@/hooks/useEnvironment";

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [reportContent, setReportContent] = useState<string | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [selectedTask, setSelectedTask] = useState('random_incident');
  const [chaosMode, setChaosMode] = useState(true);
  const [modelConfig, setModelConfig] = useState({
    base_model: 'sft_merged_1p5b_v5',
    adapter_path: 'trained_model_1p5b_v5',
    device: 'auto' as const,
  });
  
  const {
    state, isConnected, error, rewardHistory, resetEnvironment,
    isAutoPilotRunning, startAutoPilot, stopAutoPilot, autoPilotSteps,
    liveScore, fetchLiveScore,
    modelInfo, fetchModelInfo,
  } = useEnvironment(800);

  // Tiny toast/banner for chaos: show latest event (persisted) and pulse when new
  const chaosEvent = (state?.metadata?.last_chaos_event as string | undefined) || (state?.metadata?.new_chaos_event as string | undefined);
  const chaosIsNew = Boolean(state?.metadata?.new_chaos_event);

  // Auto-fetch report when episode resolves
  const fetchReport = useCallback(async () => {
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '') || 'http://localhost:8000';
      const res = await fetch(`${apiBase}/report`);
      if (res.ok) {
        const data = await res.json();
        setReportContent(data.report);
        setShowReport(true);
      }
    } catch (e) {
      console.error('Failed to fetch report:', e);
    }
  }, []);

  // When episode resolves, auto-show report
  useEffect(() => {
    if (state?.is_resolved && !isAutoPilotRunning) {
      fetchReport();
      fetchLiveScore();
    }
  }, [state?.is_resolved, isAutoPilotRunning, fetchReport, fetchLiveScore]);

  // Handle window resize logic for responsive sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      }
    };

    // Initial check
    handleResize();

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleToggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  const handleReset = async (taskName: string, chaosMode: boolean) => {
    setReportContent(null);
    setShowReport(false);
    await resetEnvironment(taskName, chaosMode);
  };

  const ensureReset = useCallback(async () => {
    if (!state?.episode_id) {
      await resetEnvironment(selectedTask, chaosMode);
    }
  }, [state?.episode_id, resetEnvironment, selectedTask, chaosMode]);

  const handleStartAutoPilot = useCallback(() => {
    startAutoPilot({ ensureReset, modelConfig });
  }, [startAutoPilot, ensureReset, modelConfig]);

  return (
    <div className="flex w-full h-full min-h-screen">
      <Sidebar
        collapsed={sidebarCollapsed}
        state={state}
        selectedTask={selectedTask}
        onSelectedTaskChange={setSelectedTask}
        chaosMode={chaosMode}
        onChaosModeChange={setChaosMode}
        modelConfig={modelConfig}
        onModelConfigChange={setModelConfig}
        modelInfo={modelInfo}
        onReset={handleReset}
        isAutoPilotRunning={isAutoPilotRunning}
        onStartAutoPilot={handleStartAutoPilot}
        onStopAutoPilot={stopAutoPilot}
        liveScore={liveScore}
      />

      <div
        className={`flex-1 flex flex-col min-h-screen main-content-transition ${sidebarCollapsed ? 'expanded-main' : 'ml-60'}`}
        id="main-wrapper"
      >
        <Header onToggleSidebar={handleToggleSidebar} isConnected={isConnected} taskName={state?.task_name} />

        <main className="flex-1 pt-8 px-8 pb-12 overflow-y-auto">
          <div className="mb-stack-lg max-w-[1400px] mx-auto w-full flex justify-between items-end">
            <div>
              <h2 className="font-h1 text-h1 text-on-surface">Service Map Overview</h2>
              <p className="font-body-md text-body-md text-on-surface-variant mt-1">Real-time status of interconnected core systems.</p>
            </div>
            <div className="flex items-center gap-3">
              {chaosEvent && (
                <div className={`flex items-center gap-2 bg-fuchsia-500/15 text-fuchsia-200 px-4 py-2 rounded-lg border border-fuchsia-500/30 font-body-sm ${chaosIsNew ? 'animate-pulse' : ''}`}>
                  <span className="material-symbols-outlined text-sm">bolt</span>
                  Chaos event: <span className="font-mono">{chaosEvent}</span>
                </div>
              )}
              {/* Resolved Banner */}
              {state?.is_resolved && (
                <div className="flex items-center gap-2 bg-emerald-500/15 text-emerald-300 px-4 py-2 rounded-lg border border-emerald-500/30 font-body-sm animate-pulse">
                  <span className="material-symbols-outlined text-sm">check_circle</span>
                  Incident Resolved
                  <button
                    onClick={fetchReport}
                    className="ml-2 text-[10px] font-bold uppercase bg-emerald-500/20 hover:bg-emerald-500/30 px-2 py-0.5 rounded transition-colors"
                  >
                    View Report
                  </button>
                </div>
              )}
              {error && (
                <div className="bg-error/20 text-error px-4 py-2 rounded-lg border border-error/50 font-body-sm">
                  Connection Error: {error}
                </div>
              )}
            </div>
          </div>

          <div className="responsive-dashboard-grid max-w-[1400px] mx-auto w-full">
            <div className="grid-col-span-core flex flex-col gap-stack-md min-w-0">
              <ServiceMap services={state?.services || {}} />
            </div>

            <div className="grid-col-span-activity flex flex-col gap-stack-md min-w-0">
              <ActivityLog
                timeline={state?.incident_timeline || []}
                runbookMemory={state?.runbook_memory}
                escalationTier={state?.escalation_tier}
                servicesAtRisk={state?.services_at_risk}
              />
            </div>

            <div className="grid-col-span-metrics flex flex-col gap-stack-md min-w-0">
              <MetricsGauges services={state?.services || {}} />
            </div>

            <div className="grid-col-span-perf flex flex-col gap-stack-md min-w-0 mt-4">
              <PerformanceTrend rewardHistory={rewardHistory} />
            </div>
          </div>
        </main>
      </div>

      {/* Post-Incident Report Modal */}
      <ReportModal
        isOpen={showReport}
        onClose={() => setShowReport(false)}
        reportContent={reportContent}
      />
    </div>
  );
}

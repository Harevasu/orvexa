import React, { useState } from 'react';
import { Header } from './components/Header';
import { NavTab, Sidebar } from './components/Sidebar';
import { ConjunctionAnalysis } from './pages/ConjunctionAnalysis';
import { EventDetail } from './pages/EventDetail';
import { Explanations } from './pages/Explanations';
import { HorizonComparison } from './pages/HorizonComparison';
import { OrbitalDemo } from './pages/OrbitalDemo';
import { RankedAlerts } from './pages/RankedAlerts';
import { Reliability } from './pages/Reliability';
import { Robustness } from './pages/Robustness';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('analysis');
  const [activeHorizon, setActiveHorizon] = useState<string>('H2');
  const [selectedEventId, setSelectedEventId] = useState<string>('6709');

  const handleSelectEventAndNavigate = (eventId: string) => {
    setSelectedEventId(eventId);
    setActiveTab('analysis');
  };

  const handleNavigateToDetail = () => {
    setActiveTab('event_detail');
  };

  return (
    <div className="min-h-screen bg-space-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-space-950">
      {/* Top Navigation Bar */}
      <Header
        activeHorizon={activeHorizon}
        onHorizonChange={setActiveHorizon}
      />

      {/* Main Layout Body */}
      <div className="flex-1 flex flex-col md:flex-row max-w-7xl w-full mx-auto">
        {/* Left Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          selectedEventId={selectedEventId}
        />

        {/* Content View Area */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto max-w-full">
          {activeTab === 'analysis' && (
            <ConjunctionAnalysis
              activeHorizon={activeHorizon}
              onHorizonChange={setActiveHorizon}
              selectedEventId={selectedEventId}
              onSelectEvent={setSelectedEventId}
              onNavigateToDetail={handleNavigateToDetail}
            />
          )}

          {activeTab === 'event_detail' && (
            <EventDetail
              eventId={selectedEventId}
              onBackToAnalysis={() => setActiveTab('analysis')}
            />
          )}

          {activeTab === 'horizon_comparison' && (
            <HorizonComparison
              selectedEventId={selectedEventId}
              onSelectHorizon={(h) => {
                setActiveHorizon(h);
                setActiveTab('analysis');
              }}
            />
          )}

          {activeTab === 'ranked_alerts' && (
            <RankedAlerts
              activeHorizon={activeHorizon}
              onHorizonChange={setActiveHorizon}
              onSelectEventAndNavigate={handleSelectEventAndNavigate}
            />
          )}

          {activeTab === 'reliability' && <Reliability />}

          {activeTab === 'robustness' && <Robustness />}

          {activeTab === 'explanations' && <Explanations />}

          {activeTab === 'orbital_demo' && <OrbitalDemo />}
        </main>
      </div>

      {/* Persistent Footer */}
      <footer className="bg-space-900 border-t border-slate-800/80 py-3 px-4 text-center text-xs text-slate-500 font-mono">
        ORVEXA: Deep Conformal Risk Prioritization for Orbital Conjunctions | Final Year Project Demonstration
      </footer>
    </div>
  );
};

export default App;

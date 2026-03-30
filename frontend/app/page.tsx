'use client';

import React, { useState, useEffect } from 'react';
import SearchBox from '@/components/SearchBox';
import ResultItem from '@/components/ResultItem';
import SummaryBox from '@/components/SummaryBox';
import Header from '@/components/Header';
import { Settings, Sparkles, Loader2, Info } from 'lucide-react';
import { useRouter, useSearchParams } from 'next/navigation';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

const RECOMMENDATIONS = [
  "Stundenplan",
  "Prüfungsordnung",
  "Vorlesungsverzeichnis",
  "Modulhandbuch",
  "Klausurtermine",
  "Bibliothek",
  "Mensa",
  "Studienberatung",
  "Bewerbung",
  "Notenspiegel"
];

interface SearchResult {
  type: 'webpage' | 'timetable' | 'website' | 'asta' | 'pdf';
  title: string;
  url: string;
  text: string;
  program?: string;
  day?: string;
  time?: string;
  room?: string;
  semester?: string;
  score: number;
}

interface SearchResponse {
  results: SearchResult[];
  summary?: string;
  total_results: number;
  original_query: string;
  expanded_query: string;
  semester: string;
  llm_enabled: boolean;
  sources?: { index: number; url: string }[];
  model?: string;
  provider?: string;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [summary, setSummary] = useState<string | undefined>(undefined);
  const [sources, setSources] = useState<{ index: number; url: string }[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLanding, setIsLanding] = useState(true);
  const [totalResults, setTotalResults] = useState(0);
  const [responseTime, setResponseTime] = useState(0);
  const [llmEnabled, setLlmEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState('all');
  
  const [showOptions, setShowOptions] = useState(false);
  const [logoClicks, setLogoClicks] = useState(0);
  const [currentModel, setCurrentModel] = useState<string | undefined>(undefined);
  const [currentProvider, setCurrentProvider] = useState<string | undefined>(undefined);
  const [loadingSummary, setLoadingSummary] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl + Shift + O to toggle options
      if (e.ctrlKey && e.shiftKey && e.key === 'O') {
        setShowOptions(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleLogoClick = () => {
    setIsLanding(true);
    setLogoClicks(prev => {
      const next = prev + 1;
      if (next >= 5) {
        setShowOptions(p => !p);
        return 0;
      }
      return next;
    });
  };

  const [options, setOptions] = useState({
    provider: 'github',
    model: 'openai/gpt-5',
    semester: 'SoSe26',
    strict: true,
    summary: true,
    rerank: true,
    expansion: true,
    openaiKey: ''
  });

  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    fetchModels();
  }, [options.provider, options.openaiKey]);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/models?provider=${options.provider}`, {
        headers: { 'X-OpenAI-Key': options.openaiKey }
      });
      const data = await res.json();
      setModels(data.models || []);
      if (data.models && data.models.length > 0 && !options.model) {
        setOptions(prev => ({ ...prev, model: data.models[0] }));
      }
    } catch (err) {
      console.error('Failed to fetch models', err);
    }
  };

  const router = useRouter();
  const searchParams = useSearchParams();

  // Verlauf (max. 10 Einträge)
  const [history, setHistory] = useState<string[]>([]);

  // Lade Verlauf aus LocalStorage beim Start
  useEffect(() => {
    const stored = localStorage.getItem('searchHistory');
    if (stored) setHistory(JSON.parse(stored));
  }, []);

  // Schreibe Verlauf in LocalStorage, wenn er sich ändert
  useEffect(() => {
    localStorage.setItem('searchHistory', JSON.stringify(history));
  }, [history]);

  // URL-Query übernehmen (z.B. bei Back/Forward)
  useEffect(() => {
    const q = searchParams.get('q') || '';
    if (q && q !== query) {
      setQuery(q);
      handleSearch(q, false); // false: nicht erneut pushen
    }
    // eslint-disable-next-line
  }, [searchParams]);

  const handleSearch = async (val: string, pushState = true) => {
    if (!val.trim()) return;
    setQuery(val);
    setIsLoading(true);
    setIsLanding(false);
    setError(null);
    setActiveFilter('all');
    const start = performance.now();
    if (pushState) {
      router.push(`/?q=${encodeURIComponent(val)}`);
    }
    // Verlauf aktualisieren
    setHistory(prev => {
      const arr = [val, ...prev.filter(v => v !== val)];
      return arr.slice(0, 10);
    });
    try {
      const params = new URLSearchParams({
        q: val,
        include_summary: options.summary.toString(),
        include_rerank: options.rerank.toString(),
        include_expansion: options.expansion.toString(),
        strict_match: options.strict.toString(),
        semester: options.semester,
        model_name: options.model,
        provider: options.provider
      });

      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/search?${params}`, {
        headers: { 'X-OpenAI-Key': options.openaiKey }
      });

      if (res.status === 429) {
        const data = await res.json();
        setError(data.detail || "Limit erreicht. Bitte später erneut versuchen.");
        setResults([]);
        setSummary(undefined);
        setIsLoading(false);
        return;
      }

      if (!res.ok) {
        throw new Error(`Search failed: ${res.status}`);
      }

      const data: SearchResponse = await res.json();
      
      setResults(data.results);
      setSummary(data.summary); // Use summary from search!
      setTotalResults(data.total_results);
      setSources(data.sources || []);
      setLlmEnabled(data.llm_enabled);
      setResponseTime(Math.round(performance.now() - start));
      setCurrentModel(data.model);
      setCurrentProvider(data.provider);
      setIsLoading(false);

      // Only fetch summary if search didn't provide one and it's enabled
      if (options.summary && !data.summary && data.results.length > 0) {
        setLoadingSummary(true);
        try {
          const sumRes = await fetch(`${API_BASE_URL}/api/summarize`, {
            method: 'POST',
            headers: { 
              'Content-Type': 'application/json',
              'X-OpenAI-Key': options.openaiKey 
            },
            body: JSON.stringify({
              q: val,
              results: data.results,
              provider: options.provider,
              model_name: options.model
            })
          });
          
          if (sumRes.ok) {
            const sumData = await sumRes.json();
            setSummary(sumData.summary);
            setCurrentModel(sumData.model);
            setCurrentProvider(sumData.provider);
            
            if (sumData.summary && sumData.summary.startsWith("ERROR: LIMIT_EXCEEDED")) {
              setError("KI-Limit erreicht. Die Zusammenfassung ist derzeit nicht verfügbar.");
              setSummary(undefined);
            }
          }
        } catch (sumErr) {
          console.error('Summary fetch failed', sumErr);
        } finally {
          setLoadingSummary(false);
        }
      }
    } catch (err) {
      console.error('Search failed', err);
      setError("Ein unerwarteter Fehler ist aufgetreten.");
      setIsLoading(false);
    }
  };

  const handleFeedback = async (rating: 1 | -1) => {
    try {
      await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          summary: summary,
          rating: rating,
          model: options.model
        })
      });
    } catch (err) {
      console.error('Feedback failed', err);
    }
  };

  if (isLanding) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center p-5 bg-[var(--bg)]">
        <div 
          onClick={handleLogoClick}
          className="mb-10 text-center select-none animate-in fade-in zoom-in duration-500 cursor-pointer active:scale-95 transition-transform"
        >
          <h1 className="text-6xl font-bold tracking-tighter sm:text-7xl">
            <span className="text-[#8ab4f8]">HS</span>
            <span className="text-[#f28b82]">.</span>
            <span className="text-[#81c995]">Aalen</span>
            <span className="text-[#fdd663] text-2xl vertical-align-super ml-2">AI</span>
          </h1>
        </div>

        <SearchBox onSearch={handleSearch} isLanding={true} history={history} recommendations={RECOMMENDATIONS} />
        {/* Verlaufsliste */}
        {history.length > 0 && (
          <div className="mt-4 w-full max-w-[584px] text-left">
            <div className="text-xs text-[var(--text-secondary)] mb-1">Letzte Suchen:</div>
            <div className="flex flex-wrap gap-2">
              {history.map((h, i) => (
                <button
                  key={h + i}
                  onClick={() => handleSearch(h)}
                  className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded text-xs hover:bg-[var(--accent)] hover:text-white transition-colors"
                >
                  {h}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <button 
            onClick={() => handleSearch(query)}
            className="px-6 py-2.5 bg-[var(--surface)] text-[var(--text)] border border-[var(--surface)] hover:border-[var(--border)] rounded-md transition-colors text-sm font-medium"
          >
            AI-Suche
          </button>
          
          {showOptions && (
            <div className="relative group">
              <button className="px-6 py-2.5 bg-[var(--surface)] text-[var(--text)] border border-[var(--border)] hover:border-[var(--border)] rounded-md transition-colors text-sm font-medium flex items-center gap-2">
                <Settings className="w-4 h-4" /> Optionen
              </button>
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-[min(440px,92vw)] bg-[var(--surface)] border border-[var(--border)] rounded-xl p-5 shadow-2xl z-50 invisible group-hover:visible transition-all opacity-0 group-hover:opacity-100">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-[0.75rem] text-[var(--text-secondary)] mb-1.5 block">Provider</label>
                      <select 
                        value={options.provider}
                        onChange={(e) => setOptions(prev => ({ ...prev, provider: e.target.value }))}
                        className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg p-2 text-sm outline-none focus:border-[var(--accent)]"
                      >
                        <option value="github">Auto (GitHub)</option>
                        <option value="none">Keine KI</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[0.75rem] text-[var(--text-secondary)] mb-1.5 block">Modell</label>
                      <select 
                        value={options.model}
                        onChange={(e) => setOptions(prev => ({ ...prev, model: e.target.value }))}
                        className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg p-2 text-sm outline-none focus:border-[var(--accent)]"
                      >
                        {models.map(m => <option key={m} value={m}>{m}</option>)}
                        {models.length === 0 && <option value="">Modelle laden...</option>}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 border-t border-[var(--border)] pt-4">
                    <label className="flex items-center gap-2 text-sm cursor-pointer select-none text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors">
                        <input 
                          type="checkbox" 
                          checked={options.summary} 
                          onChange={(e) => setOptions(prev => ({ ...prev, summary: e.target.checked }))}
                          className="w-4 h-4 accent-[var(--accent)]"
                        /> Zusammenfassung
                    </label>
                    <label className="flex items-center gap-2 text-sm cursor-pointer select-none text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors">
                        <input 
                          type="checkbox" 
                          checked={options.strict} 
                          onChange={(e) => setOptions(prev => ({ ...prev, strict: e.target.checked }))}
                          className="w-4 h-4 accent-[var(--accent)]"
                        /> Strikte Suche
                    </label>
                    <label className="flex items-center gap-2 text-sm cursor-pointer select-none text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors">
                        <input 
                          type="checkbox" 
                          checked={options.rerank} 
                          onChange={(e) => setOptions(prev => ({ ...prev, rerank: e.target.checked }))}
                          className="w-4 h-4 accent-[var(--accent)]"
                        /> Re-Ranking
                    </label>
                    <label className="flex items-center gap-2 text-sm cursor-pointer select-none text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors">
                        <input 
                          type="checkbox" 
                          checked={options.expansion} 
                          onChange={(e) => setOptions(prev => ({ ...prev, expansion: e.target.checked }))}
                          className="w-4 h-4 accent-[var(--accent)]"
                        /> KI-Erweiterung
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-12 text-[var(--text-secondary)] text-sm max-w-2xl text-center leading-relaxed opacity-70">
          Transparenz: Die KI erstellt nur die Zusammenfassung und hilft beim Sortieren. Erfahre mehr über unsere Quellen in den internen Einstellungen.
        </div>
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col">
      <header className="bg-[var(--surface)] border-b border-[var(--border)] sticky top-0 z-50 flex flex-col pt-4">
        <div className="px-5 flex items-center gap-6 mb-2">
          <div 
            onClick={handleLogoClick}
            className="text-2xl font-bold tracking-tighter cursor-pointer select-none shrink-0 active:scale-95 transition-transform"
          >
            <span className="text-[#8ab4f8]">HS</span>
            <span className="text-[#f28b82]">.</span>
            <span className="text-[#81c995]">Aalen</span>
          </div>
          <div className="flex-1 max-w-[692px]">
            <SearchBox onSearch={handleSearch} initialValue={query} history={history} recommendations={RECOMMENDATIONS} />
          </div>
        </div>
        
        <div className="px-5 lg:pl-[184px]">
          <div className="flex items-center gap-6 border-b border-[var(--border)] mb-6 scrollbar-hide overflow-x-auto">
            <button 
              onClick={() => setActiveFilter('all')}
              className={`pb-3 text-sm font-medium transition-all relative ${activeFilter === 'all' ? 'text-[var(--accent)]' : 'text-[var(--text-secondary)] hover:text-[var(--text)]'}`}
            >
              Alle
              {activeFilter === 'all' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent)]" />}
            </button>
            <button 
              onClick={() => setActiveFilter('timetable')}
              className={`pb-3 text-sm font-medium transition-all relative ${activeFilter === 'timetable' ? 'text-[var(--accent)]' : 'text-[var(--text-secondary)] hover:text-[var(--text)]'}`}
            >
              Vorlesungen
              {activeFilter === 'timetable' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent)]" />}
            </button>
            <button 
              onClick={() => setActiveFilter('website')}
              className={`pb-3 text-sm font-medium transition-all relative ${activeFilter === 'website' ? 'text-[var(--accent)]' : 'text-[var(--text-secondary)] hover:text-[var(--text)]'}`}
            >
              Webseiten
              {activeFilter === 'website' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent)]" />}
            </button>
            <button 
              onClick={() => setActiveFilter('pdf')}
              className={`pb-3 text-sm font-medium transition-all relative ${activeFilter === 'pdf' ? 'text-[var(--accent)]' : 'text-[var(--text-secondary)] hover:text-[var(--text)]'}`}
            >
              PDFs
              {activeFilter === 'pdf' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent)]" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-[1200px] mx-auto px-5 lg:pl-[120px] py-6">
        {isLoading ? (
          <div className="mt-10 space-y-8 animate-pulse">
            <div className="h-4 bg-[var(--surface)] w-40 rounded mb-4" />
            <div className="h-[200px] bg-[var(--surface)] rounded-2xl w-full max-w-[692px]" />
            {[1, 2, 3].map(i => (
              <div key={i} className="space-y-3 max-w-[692px]">
                <div className="h-4 bg-[var(--surface)] w-1/4 rounded" />
                <div className="h-6 bg-[var(--surface)] w-3/4 rounded" />
                <div className="h-20 bg-[var(--surface)] w-full rounded" />
              </div>
            ))}
          </div>
        ) : (
          <div className="w-full max-w-[692px]">
            <div className="text-[0.8rem] text-[var(--text-secondary)] mb-6 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Info className="w-3.5 h-3.5" />
                Ungefähr {totalResults} Ergebnisse · in {responseTime}ms · Semester: {options.semester}
              </div>
              {!llmEnabled && (
                <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[var(--surface)] border border-[var(--border)] rounded-full text-[0.7rem] font-medium text-[var(--text-secondary)]">
                  <Sparkles className="w-3 h-3 opacity-50" /> Reine Vektor-Suche
                </div>
              )}
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-500 text-sm flex items-center gap-3">
                <Info className="w-5 h-5 shrink-0" />
                {error}
              </div>
            )}

            {(summary || loadingSummary) && activeFilter === 'all' && (
              <SummaryBox 
                summary={summary || ""} 
                sources={sources} 
                model={currentModel} 
                provider={currentProvider}
                onFeedback={handleFeedback}
                loading={loadingSummary}
              />
            )}

            <div className="space-y-2">
              {results
                .filter(r => {
                  if (activeFilter === 'all') return true;
                  if (activeFilter === 'timetable') return r.type === 'timetable';
                  if (activeFilter === 'pdf') return r.type === 'pdf' || r.url?.toLowerCase().endsWith('.pdf');
                  if (activeFilter === 'website') return ['website', 'webpage', 'asta'].includes(r.type) && !r.url?.toLowerCase().endsWith('.pdf');
                  return true;
                })
                .map((result, i) => (
                  <ResultItem key={`${result.url}-${i}`} result={result} query={query} />
                ))}
            </div>
          </div>
        )}
      </main>

      <footer className="mt-auto border-t border-[var(--border)] bg-[var(--surface)] p-8">
        <div className="max-w-[1200px] mx-auto px-5 lg:pl-[120px] flex flex-col md:flex-row justify-between gap-8 text-[var(--text-secondary)] text-sm">
           <div className="space-y-4 max-w-md">
              <h4 className="text-[var(--text)] font-semibold uppercase tracking-wider text-xs">Über diesen Dienst</h4>
              <p className="leading-relaxed opacity-80">
                Diese Suchmaschine ist ein KI-gestütztes Tool für Studierende der Hochschule Aalen. 
                Sie indexiert offizielle Webseiten, PDFs und Prüfungsordnungen.
              </p>
           </div>
           <div className="grid grid-cols-2 gap-x-12 gap-y-2">
              <h4 className="col-span-2 text-[var(--text)] font-semibold uppercase tracking-wider text-xs mb-2">Rechtliches</h4>
              <a href="#" className="hover:text-[var(--accent)] transition-colors">Impressum</a>
              <a href="#" className="hover:text-[var(--accent)] transition-colors">Datenschutz</a>
              <a href="#" className="hover:text-[var(--accent)] transition-colors">Kontakt</a>
              <a href="#" className="hover:text-[var(--accent)] transition-colors">Lizenzen</a>
           </div>
        </div>
        <div className="mt-8 pt-8 border-t border-[var(--border)] flex flex-col items-center gap-2 text-[var(--text-secondary)] text-[0.7rem] opacity-60 uppercase tracking-[0.2em]">
           <div>HS Aalen Search v2.0 • Powered by Next.js & FastAPI</div>
           {(currentModel || options.provider !== 'none') && (
             <div className="flex items-center gap-2 lowercase tracking-normal opacity-100 mt-1">
               <Sparkles className="w-3 h-3 text-[var(--accent)]" />
               <span>AI Status: {currentModel || 'Suche...'} ({currentProvider || options.provider})</span>
             </div>
           )}
        </div>
      </footer>
    </div>
  );
}

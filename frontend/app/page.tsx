'use client';

import React, { useState, useEffect, Suspense } from 'react';
import SearchBox from '@/components/SearchBox';
import ResultItem from '@/components/ResultItem';
import Header from '@/components/Header';
import { Info } from 'lucide-react';
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
  total_results: number;
  filters?: { intent: string; entity: string };
}

function SearchContent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLanding, setIsLanding] = useState(true);
  const [totalResults, setTotalResults] = useState(0);
  const [responseTime, setResponseTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const router = useRouter();
  const searchParams = useSearchParams();

  // Verlauf (max. 10 Einträge)
  const [history, setHistory] = useState<string[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>(RECOMMENDATIONS);

  // Lade dynamische Vorschläge vom Backend
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/suggestions`)
      .then(res => res.json())
      .then(data => {
        if (data.suggestions && data.suggestions.length > 0) {
          // Kombiniere populäre Community-Suchen (vorne) mit statischen (hinten), max 15
          const merged = Array.from(new Set([...data.suggestions, ...RECOMMENDATIONS])).slice(0, 15);
          setRecommendations(merged);
        }
      })
      .catch(err => console.error("Failed to fetch suggestions", err));
  }, []);

  // Lade Verlauf aus LocalStorage beim Start
  useEffect(() => {
    const stored = localStorage.getItem('searchHistory');
    if (stored) setHistory(JSON.parse(stored));
  }, []);

  // Schreibe Verlauf in LocalStorage, wenn er sich ändert
  useEffect(() => {
    localStorage.setItem('searchHistory', JSON.stringify(history));
  }, [history]);

  // Long click Tracking (NavBoost)
  useEffect(() => {
    const handleVisibilityChange = () => {
      // Erkennen, wenn User zum Such-Tab zurückkehrt
      if (document.visibilityState === 'visible' || document.hasFocus()) {
        const lastUrl = sessionStorage.getItem('lastClickedUrl');
        const lastQuery = sessionStorage.getItem('lastClickedQuery');
        const timeStr = sessionStorage.getItem('lastClickedTime');
        
        if (lastUrl && lastQuery && timeStr) {
          const timeSpent = Date.now() - parseInt(timeStr, 10);
          
          // Wenn User 20+ Sekunden weg war = Long Click
          if (timeSpent > 20000) {
            fetch(`${API_BASE_URL}/api/feedback/click`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ query: lastQuery, url: lastUrl, type: 'long_click' })
            }).catch(() => {});
          }
          
          sessionStorage.removeItem('lastClickedUrl');
          sessionStorage.removeItem('lastClickedQuery');
          sessionStorage.removeItem('lastClickedTime');
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleVisibilityChange);
    };
  }, []);

  // URL-Query übernehmen (z.B. bei Back/Forward)
  useEffect(() => {
    const q = searchParams.get('q') || '';
    const p = parseInt(searchParams.get('page') || '1', 10);
    if (q) {
      if (q !== query || p !== currentPage) {
        setQuery(q);
        setCurrentPage(p);
        handleSearch(q, p, false);
      }
    }
    // eslint-disable-next-line
  }, [searchParams]);

  const handleSearch = async (val: string, page: number = 1, pushState = true) => {
    if (!val.trim()) return;
    setQuery(val);
    setCurrentPage(page);
    setIsLoading(true);
    setIsLanding(false);
    setError(null);
    setActiveFilter('all');
    const start = performance.now();
    if (pushState) {
      router.push(`/?q=${encodeURIComponent(val)}&page=${page}`);
    }
    // Verlauf aktualisieren (nur bei neuen Suchen)
    if (page === 1) {
      setHistory(prev => {
        const arr = [val, ...prev.filter(v => v !== val)];
        return arr.slice(0, 10);
      });
    }
    try {
      const params = new URLSearchParams({ q: val, page: page.toString() });

      const res = await fetch(`${API_BASE_URL}/api/search?${params}`);

      if (res.status === 429) {
        const data = await res.json();
        setError(data.detail || "Limit erreicht. Bitte später erneut versuchen.");
        setResults([]);
        setIsLoading(false);
        return;
      }

      if (!res.ok) {
        throw new Error(`Search failed: ${res.status}`);
      }

      const data = await res.json();
      
      setResults(data.results);
      setTotalResults(data.total_results || 0);
      setTotalPages(data.total_pages || 1);
      setCurrentPage(data.page || 1);
      setResponseTime(Math.round(performance.now() - start));
      setIsLoading(false);
    } catch (err) {
      console.error('Search failed', err);
      setError("Ein unerwarteter Fehler ist aufgetreten.");
      setIsLoading(false);
    }
  };

  if (isLanding) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center p-5 bg-[var(--bg)]">
        <div 
          onClick={() => setIsLanding(true)}
          className="mb-10 text-center select-none animate-in fade-in zoom-in duration-500 cursor-pointer active:scale-95 transition-transform"
        >
          <h1 className="text-6xl font-bold tracking-tighter sm:text-7xl">
            <span className="text-[#8ab4f8]">HS</span>
            <span className="text-[#f28b82]">.</span>
            <span className="text-[#81c995]">Aalen</span>
          </h1>
        </div>

        <SearchBox onSearch={handleSearch} isLanding={true} history={history} recommendations={recommendations} />
        <div className="mt-8 flex flex-wrap justify-center gap-3 animate-in fade-in zoom-in duration-500 delay-100 fill-mode-both">
          <button 
            onClick={() => handleSearch(query)}
            className="px-6 py-2.5 bg-[var(--surface)] text-[var(--text)] border border-[var(--surface)] hover:border-[var(--border)] rounded-md transition-all shadow-sm hover:shadow text-sm font-medium"
          >
            Suche starten
          </button>
        </div>

        {/* Verlaufsliste */}
        {history.length > 0 && (
          <div className="mt-8 w-full max-w-[600px] flex flex-col items-center animate-in fade-in slide-in-from-bottom-2 duration-500 delay-200 fill-mode-both">
            <div className="flex flex-wrap items-center justify-center gap-2.5">
              {history.slice(0, 5).map((h, i) => (
                <button
                  key={h + i}
                  onClick={() => handleSearch(h)}
                  className="group flex items-center gap-1.5 px-3 py-1.5 bg-[var(--surface)] hover:bg-[var(--accent)] border border-[var(--border)] rounded-full text-xs text-[var(--text-secondary)] hover:text-white transition-all duration-300 shadow-sm hover:shadow-md cursor-pointer"
                >
                  <svg className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 transition-opacity" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  {h}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-12 text-[var(--text-secondary)] text-sm max-w-2xl text-center leading-relaxed opacity-70">
          Durchsuche offizielle Webseiten, PDFs, Prüfungsordnungen und Vorlesungspläne der Hochschule Aalen.
        </div>
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col">
      <header className="bg-[var(--surface)] border-b border-[var(--border)] sticky top-0 z-50 flex flex-col pt-4">
        <div className="px-5 flex items-center gap-6 mb-2">
          <div 
            onClick={() => setIsLanding(true)}
            className="text-2xl font-bold tracking-tighter cursor-pointer select-none shrink-0 active:scale-95 transition-transform"
          >
            <span className="text-[#8ab4f8]">HS</span>
            <span className="text-[#f28b82]">.</span>
            <span className="text-[#81c995]">Aalen</span>
          </div>
          <div className="flex-1 max-w-[692px]">
            <SearchBox onSearch={handleSearch} initialValue={query} history={history} recommendations={recommendations} />
          </div>
        </div>
        
        <div className="px-5 lg:pl-[120px]">
          <div className="flex items-center gap-8 scrollbar-hide overflow-x-auto">
            <button 
              onClick={() => setActiveFilter('all')}
              className={`tab-btn ${activeFilter === 'all' ? 'tab-btn-active' : 'tab-btn-inactive'}`}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              Alle
              {activeFilter === 'all' && <div className="tab-underline" />}
            </button>
            <button 
              onClick={() => setActiveFilter('timetable')}
              className={`tab-btn ${activeFilter === 'timetable' ? 'tab-btn-active' : 'tab-btn-inactive'}`}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
              Vorlesungen
              {activeFilter === 'timetable' && <div className="tab-underline" />}
            </button>
            <button 
              onClick={() => setActiveFilter('website')}
              className={`tab-btn ${activeFilter === 'website' ? 'tab-btn-active' : 'tab-btn-inactive'}`}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
              Webseiten
              {activeFilter === 'website' && <div className="tab-underline" />}
            </button>
            <button 
              onClick={() => setActiveFilter('pdf')}
              className={`tab-btn ${activeFilter === 'pdf' ? 'tab-btn-active' : 'tab-btn-inactive'}`}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              PDFs
              {activeFilter === 'pdf' && <div className="tab-underline" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-[1200px] mx-auto px-5 lg:pl-[120px] py-6">
        {isLoading ? (
          <div className="mt-10 space-y-8 animate-pulse">
            <div className="h-4 bg-[var(--surface)] w-40 rounded mb-4" />
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
            <div className="text-[0.8rem] text-[var(--text-secondary)] mb-6 flex items-center gap-2">
              <Info className="w-3.5 h-3.5" />
              Ungefähr {totalResults} Ergebnisse · in {responseTime}ms
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-500 text-sm flex items-center gap-3">
                <Info className="w-5 h-5 shrink-0" />
                {error}
              </div>
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

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-6 mt-12 mb-8">
                <button
                  disabled={currentPage <= 1}
                  onClick={() => {
                    handleSearch(query, currentPage - 1);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                  className="px-5 py-2.5 border border-[var(--border)] rounded-md text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--surface)] hover:text-[var(--text)] transition-all"
                >
                  Zurück
                </button>
                <span className="text-sm font-medium text-[var(--text-secondary)]">Seite {currentPage} von {totalPages}</span>
                <button
                  disabled={currentPage >= totalPages}
                  onClick={() => {
                    handleSearch(query, currentPage + 1);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                  className="px-5 py-2.5 border border-[var(--border)] rounded-md text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--surface)] hover:text-[var(--text)] transition-all"
                >
                  Weiter
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="mt-auto border-t border-[var(--border)] bg-[var(--surface)] p-8">
        <div className="max-w-[1200px] mx-auto px-5 lg:pl-[120px] flex flex-col md:flex-row justify-between gap-8 text-[var(--text-secondary)] text-sm">
           <div className="space-y-4 max-w-md">
              <h4 className="text-[var(--text)] font-semibold uppercase tracking-wider text-xs">Über diesen Dienst</h4>
              <p className="leading-relaxed opacity-80">
                Diese Suchmaschine ist ein Vektorsuche-Tool für Studierende der Hochschule Aalen. 
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
        </div>
      </footer>
    </div>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen flex flex-col items-center justify-center p-5 bg-[var(--bg)]"><div className="animate-pulse text-[var(--text-secondary)]">Laden...</div></div>}>
      <SearchContent />
    </Suspense>
  );
}

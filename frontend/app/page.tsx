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
    if (q && q !== query) {
      setQuery(q);
      handleSearch(q, false);
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
      const params = new URLSearchParams({ q: val });

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

      const data: SearchResponse = await res.json();
      
      setResults(data.results);
      setTotalResults(data.total_results);
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
            Suche starten
          </button>
        </div>

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

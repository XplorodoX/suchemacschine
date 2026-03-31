import React, { useState, useRef, useEffect } from 'react';

interface SearchBoxProps {
  onSearch: (query: string) => void;
  initialValue?: string;
  isLanding?: boolean;
  history?: string[];
  recommendations?: string[];
}

export default function SearchBox({ onSearch, initialValue = '', isLanding = false, history = [], recommendations = [] }: SearchBoxProps) {
  const [query, setQuery] = useState(initialValue);
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlighted, setHighlighted] = useState<number>(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  // Filter Vorschläge
  const filteredHistory = history.filter(h => h.toLowerCase().includes(query.toLowerCase()) && h !== query);
  const filteredRecs = recommendations.filter(r => r.toLowerCase().includes(query.toLowerCase()) && r !== query);
  const hasSuggestions = filteredHistory.length > 0 || filteredRecs.length > 0;

  useEffect(() => {
    if (query) setShowDropdown(true);
    else setShowDropdown(false);
    setHighlighted(-1);
  }, [query]);

  // Tastatursteuerung
  const allSuggestions = [...filteredHistory, ...filteredRecs];
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showDropdown || !hasSuggestions) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlighted(h => Math.min(h + 1, allSuggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlighted(h => Math.max(h - 1, 0));
    } else if (e.key === 'Enter' && highlighted >= 0) {
      e.preventDefault();
      setQuery(allSuggestions[highlighted]);
      setShowDropdown(false);
      onSearch(allSuggestions[highlighted]);
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    }
  };

  // Klick außerhalb schließt Dropdown
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!inputRef.current?.parentElement?.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setShowDropdown(false);
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className={`w-full ${isLanding ? 'max-w-[584px]' : 'max-w-[692px]'}`} autoComplete="off">
      <div className="search-bar group relative">
        <span className="text-[var(--text-secondary)] mr-3 flex-shrink-0">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </span>
        <div className="flex items-center flex-1 pr-1">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => query && setShowDropdown(true)}
            onKeyDown={handleKeyDown}
            placeholder="Suche..."
            className="flex-1 bg-transparent border-none outline-none text-[var(--text)] text-base py-2 placeholder:text-[var(--text-secondary)] min-w-0"
            autoFocus={isLanding}
          />
          
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="p-2 text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors"
              title="Löschen"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          )}
          
          <div className="h-6 w-[1px] bg-[var(--border)] mx-1 opacity-50 hidden sm:block"></div>
          
          {/* Google-Style Action Icons (Placeholders for design) */}
          <button type="button" className="p-2 text-[#4285f4] hover:bg-white/5 rounded-full transition-colors hidden sm:block" title="Sprachsuche">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="23"></line>
              <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
          </button>
          
          <button type="button" className="p-2 text-[#4285f4] hover:bg-white/5 rounded-full transition-colors hidden sm:block" title="Suche per Bild">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </button>
          
          <button type="submit" className="p-2 text-[#4285f4] hover:bg-white/5 rounded-full transition-colors" title="Suche">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          </button>
        </div>
        {/* Autocomplete Dropdown */}
        {showDropdown && hasSuggestions && (
          <div className="absolute left-0 top-full z-50 w-full bg-[var(--surface)] border border-[var(--border)] rounded-b shadow-lg mt-1 max-h-64 overflow-y-auto animate-in fade-in">
            {filteredHistory.length > 0 && (
              <div>
                <div className="px-3 pt-2 pb-1 text-xs text-[var(--text-secondary)]">Letzte Suchen</div>
                {filteredHistory.map((h, i) => (
                  <div
                    key={h}
                    className={`px-3 py-2 cursor-pointer hover:bg-[var(--accent)] hover:text-white ${highlighted === i ? 'bg-[var(--accent)] text-white' : ''}`}
                    onMouseDown={() => { setQuery(h); setShowDropdown(false); onSearch(h); }}
                  >
                    {h}
                  </div>
                ))}
              </div>
            )}
            {filteredRecs.length > 0 && (
              <div>
                <div className="px-3 pt-2 pb-1 text-xs text-[var(--text-secondary)]">Empfohlene Begriffe</div>
                {filteredRecs.map((r, i) => (
                  <div
                    key={r}
                    className={`px-3 py-2 cursor-pointer hover:bg-[var(--accent)] hover:text-white ${highlighted === (filteredHistory.length + i) ? 'bg-[var(--accent)] text-white' : ''}`}
                    onMouseDown={() => { setQuery(r); setShowDropdown(false); onSearch(r); }}
                  >
                    {r}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </form>
  );
}

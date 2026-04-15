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
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query && setShowDropdown(true)}
          onKeyDown={handleKeyDown}
          placeholder="Suche an der Hochschule Aalen..."
          className="flex-1 bg-transparent border-none outline-none text-[var(--text)] text-base py-2 placeholder:text-[var(--text-secondary)]"
          autoFocus={isLanding}
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery('')}
            className="text-[var(--text-secondary)] hover:text-[var(--text)] px-2 text-xl"
          >
            &times;
          </button>
        )}
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

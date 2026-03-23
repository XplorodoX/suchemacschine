import React, { useState } from 'react';

interface SearchBoxProps {
  onSearch: (query: string) => void;
  initialValue?: string;
  isLanding?: boolean;
}

export default function SearchBox({ onSearch, initialValue = '', isLanding = false }: SearchBoxProps) {
  const [query, setQuery] = useState(initialValue);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className={`w-full ${isLanding ? 'max-w-[584px]' : 'max-w-[692px]'}`}>
      <div className="search-bar group">
        <span className="text-[var(--text-secondary)] mr-3 flex-shrink-0">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </span>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
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
      </div>
    </form>
  );
}

import React from 'react';

interface HeaderProps {
  onHome: () => void;
}

export default function Header({ onHome }: HeaderProps) {
  return (
    <header className="bg-[var(--surface)] border-b border-[var(--border)] px-5 py-3 sticky top-0 z-50 flex items-center gap-6">
      <div 
        onClick={onHome}
        className="text-2xl font-bold tracking-tighter cursor-pointer select-none"
      >
        <span className="text-[#8ab4f8]">HS</span>
        <span className="text-[#f28b82]">.</span>
        <span className="text-[#81c995]">Aalen</span>
      </div>
    </header>
  );
}

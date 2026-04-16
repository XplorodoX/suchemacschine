import React from 'react';
import { ExternalLink, FileText, Calendar, Clock, MapPin, GraduationCap, Users, Globe, BookOpen } from 'lucide-react';

interface ResultItemProps {
  result: {
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
    pdf_sources?: { url: string; text?: string }[];
  };
  query: string;
}

export default function ResultItem({ result, query }: ResultItemProps) {
  const isTimetable = result.type === 'timetable';
  const isPdf = result.type === 'pdf' || result.url?.toLowerCase().endsWith('.pdf');
  const isAsta = result.type === 'asta' || (result.url && (result.url.includes('vs-hs-aalen.de') || result.url.includes('asta-aalen.de')));

  const handleLinkClick = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
    fetch(`${apiUrl}/api/feedback/click`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, url: result.url, type: 'short_click' })
    }).catch(() => {});

    sessionStorage.setItem('lastClickedUrl', result.url);
    sessionStorage.setItem('lastClickedQuery', query);
    sessionStorage.setItem('lastClickedTime', Date.now().toString());
  };

  const highlightText = (text: string, q: string) => {
    if (!text || !q) return text;
    const words = q.split(/\s+/).filter((w) => w.length > 2);
    if (words.length === 0) return text;

    let snippet = text;
    // Sort words by length descending to avoid partial replacements
    const sortedWords = [...words].sort((a, b) => b.length - a.length);
    
    sortedWords.forEach((word) => {
      const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(`(${escaped})`, 'gi');
      snippet = snippet.replace(re, '###HL###$1###ENDHL###');
    });
    
    const finalHtml = snippet
      .replace(/###HL###/g, '<mark class="bg-[var(--accent)]/20 text-inherit px-0.5 rounded font-medium">')
      .replace(/###ENDHL###/g, '</mark>');
    
    return <span dangerouslySetInnerHTML={{ __html: finalHtml }} />;
  };

  const getSourceInfo = (url: string, type: string) => {
    if (!url) return { name: 'Information', color: 'var(--text-secondary)', bg: 'var(--surface)', icon: <Globe className="w-3 h-3" /> };
    try {
      const parsedUrl = new URL(url);
      const host = parsedUrl.hostname;
      
      if (host.includes('vs-hs-aalen.de') || host.includes('asta-aalen.de') || type === 'asta') {
        return { 
          name: 'AStA Aalen', 
          color: '#a855f7', 
          bg: 'rgba(168, 85, 247, 0.1)', 
          icon: <Users className="w-3.5 h-3.5" />,
          label: 'Studierendenvertretung'
        };
      }
      if (host.includes('hs-aalen.de')) {
        return { 
          name: 'Hochschule Aalen', 
          color: '#3b82f6', 
          bg: 'rgba(59, 130, 246, 0.1)', 
          icon: <GraduationCap className="w-3.5 h-3.5" />,
          label: 'Offizielle Webseite'
        };
      }
      return { 
        name: host, 
        color: 'var(--text-secondary)', 
        bg: 'var(--surface)', 
        icon: <Globe className="w-3 h-3" />,
        label: 'Externe Quelle'
      };
    } catch (e) {
      return { name: 'Information', color: 'var(--text-secondary)', bg: 'var(--surface)', icon: <Globe className="w-3 h-3" /> };
    }
  };

  const getScrollUrl = (url: string, text: string, q: string) => {
    if (!url || !text || !q) return url;
    if (isTimetable) return url;

    const words = q.split(/\s+/).filter(w => w.length > 3);
    if (words.length === 0) return url;

    const firstWordIndex = text.toLowerCase().indexOf(words[0].toLowerCase());
    if (firstWordIndex === -1) return url;

    const start = Math.max(0, firstWordIndex - 20);
    const end = Math.min(text.length, firstWordIndex + 100);
    const context = text.substring(start, end).trim();
    
    const contextWords = context.split(/\s+/).filter(w => w.length > 2).slice(0, 5);
    if (contextWords.length < 2) return url;

    const fragment = encodeURIComponent(contextWords.join(' '));
    return `${url}#:~:text=${fragment}`;
  };

  const scrollUrl = getScrollUrl(result.url, result.text, query);
  const sourceInfo = getSourceInfo(result.url, result.type);

  if (isPdf) {
    return (
      <div className="mb-7 animate-in fade-in slide-in-from-bottom-2 duration-300">
        <div className="inline-flex items-center gap-1.5 mb-2 px-2.5 py-0.5 rounded-full border border-red-500/40 text-[0.72rem] text-red-200 bg-red-500/10">
          <FileText className="w-3 h-3" /> PDF-Dokument
        </div>
        
        <div className="flex items-center gap-2.5 mb-1.5">
          <div className="w-7 h-7 rounded-lg bg-red-500/10 flex items-center justify-center text-red-400">
            <FileText className="w-4 h-4" />
          </div>
          <div className="overflow-hidden">
            <div className="text-[0.82rem] font-medium text-[var(--text)]">{sourceInfo.name}</div>
            <div className="text-[0.72rem] text-[var(--text-secondary)] truncate flex items-center gap-1">
              {result.url && new URL(result.url).pathname.split('/').filter(Boolean).slice(0, -1).join(' › ')}
            </div>
          </div>
        </div>

        <h3 className="text-xl font-normal mb-1.5 text-[var(--link)] hover:underline decoration-red-500/30">
          <a href={result.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2" onClick={handleLinkClick}>
            {highlightText(result.title, query)} <ExternalLink className="w-4 h-4 opacity-50" />
          </a>
        </h3>
        <p className="text-[0.88rem] text-[var(--text-secondary)] leading-[1.6] line-clamp-3">
          {highlightText(result.text, query)}
        </p>
      </div>
    );
  }

  if (isTimetable) {
    return (
      <div className="mb-7 p-5 border border-[var(--border)] border-l-4 border-l-[var(--accent)] rounded-2xl bg-gradient-to-br from-[var(--accent-bg)] via-transparent to-transparent shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)]">
              <Calendar className="w-4.5 h-4.5" />
            </div>
            <div>
              <div className="text-[0.82rem] font-semibold text-[var(--text)]">Vorlesungsplan</div>
              <div className="text-[0.72rem] text-[var(--text-secondary)]">{result.program || 'Studiengang unbekannt'}</div>
            </div>
          </div>
          <div className="text-[0.65rem] uppercase tracking-wider font-bold px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-secondary)]">
            Starplan
          </div>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-3">
          {result.day && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-[0.78rem] font-medium text-[var(--accent)]">
              {result.day}
            </span>
          )}
          {result.time && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-[0.78rem] font-medium">
              <Clock className="w-3.5 h-3.5 opacity-60" /> {result.time}
            </span>
          )}
          {result.room && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-[0.78rem] font-medium">
              <MapPin className="w-3.5 h-3.5 opacity-60" /> {result.room}
            </span>
          )}
          {result.semester && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[var(--accent)]/10 text-[var(--accent)] text-[0.78rem] font-medium">
              <GraduationCap className="w-3.5 h-3.5" /> {result.semester}
            </span>
          )}
        </div>

        <h3 className="text-xl font-normal mb-2 text-[var(--link)] hover:underline decoration-[var(--accent)]/30">
          <a href={scrollUrl} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2" onClick={handleLinkClick}>
            {highlightText(result.title, query)} <ExternalLink className="w-4 h-4 opacity-50" />
          </a>
        </h3>
        <p className="text-[0.88rem] text-[var(--text-secondary)] leading-[1.6] line-clamp-2 italic">
          {highlightText(result.text, query)}
        </p>
        
        {result.url && (
          <div className="mt-3 pt-3 border-t border-[var(--border)]/50 flex items-center gap-2 text-[0.7rem] text-[var(--text-secondary)]">
            <Globe className="w-3 h-3" />
            <span className="truncate">Quelle: {new URL(result.url).hostname}</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`mb-8 p-1 rounded-xl transition-all ${isAsta ? 'group' : ''}`}>
      <div className="flex items-center gap-3 mb-2">
        <div 
          className="w-8 h-8 rounded-lg flex items-center justify-center text-white shrink-0 shadow-sm"
          style={{ backgroundColor: sourceInfo.color }}
        >
          {sourceInfo.icon}
        </div>
        <div className="overflow-hidden">
          <div className="flex items-center gap-2">
            <span className="text-[0.85rem] font-bold text-[var(--text)]">{sourceInfo.name}</span>
            {sourceInfo.label && (
              <span className="text-[0.65rem] px-1.5 py-0.25 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-secondary)] font-medium uppercase tracking-tight">
                {sourceInfo.label}
              </span>
            )}
          </div>
          <div className="text-[0.74rem] text-[var(--text-secondary)] truncate font-mono opacity-80">
            {result.url ? new URL(result.url).pathname.split('/').filter(Boolean).join(' › ') : ''}
          </div>
        </div>
      </div>

      <h3 className="text-xl font-normal mb-1.5 text-[var(--link)] hover:underline decoration-current/30">
        <a href={scrollUrl} target="_blank" rel="noopener noreferrer" onClick={handleLinkClick}>
          {highlightText(result.title, query)}
        </a>
      </h3>
      <p className="text-[0.88rem] text-[var(--text-secondary)] leading-[1.6] line-clamp-3">
        {highlightText(result.text, query)}
      </p>

      {result.pdf_sources && result.pdf_sources.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {result.pdf_sources.map((pdf, idx) => (
            <a 
              key={idx} 
              href={pdf.url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-[0.75rem] text-[var(--text)] hover:border-red-500/50 hover:bg-red-500/5 transition-all shadow-sm"
            >
              <FileText className="w-3.5 h-3.5 text-red-400" />
              <span className="truncate max-w-[200px]">{pdf.text || 'Zugehöriges PDF'}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

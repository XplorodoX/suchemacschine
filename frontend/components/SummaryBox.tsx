import React, { useState } from 'react';
import { Sparkles, ThumbsUp, ThumbsDown, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { marked } from 'marked';

interface SummaryBoxProps {
  summary: string;
  sources: { index: number; url: string }[];
  model?: string;
  provider?: string;
  onFeedback?: (rating: 1 | -1) => void;
}

export default function SummaryBox({ summary, sources, model, provider, onFeedback }: SummaryBoxProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const isError = summary.includes('⚠️');
  const formattedSummary = marked.parse(summary);

  const handleFeedback = (rating: 1 | -1) => {
    setFeedbackSent(true);
    if (onFeedback) {
      onFeedback(rating);
    }
  };

  return (
    <div className={`border-l-4 rounded-2xl p-5 mb-6 shadow-xl relative overflow-hidden transition-colors ${
      isError 
        ? 'bg-red-500/5 border-red-500/50 border' 
        : 'bg-[var(--summary-bg)] border-[var(--summary-border)] border'
    }`}>
      <div className="absolute top-0 left-0 right-0 h-full bg-gradient-to-br from-blue-500/10 via-green-500/5 to-transparent pointer-events-none" />
      
      <div className="flex items-center gap-2.5 mb-2.5 text-[0.85rem] text-[var(--accent)] font-semibold flex-wrap">
        <div className="flex items-center gap-2.5">
          <Sparkles className="w-4.5 h-4.5" />
          KI-Zusammenfassung
        </div>
        
        {model && (
          <span className="text-[0.65rem] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-white/50 font-normal">
            Modell: {model}
          </span>
        )}
        
        <span className="ml-auto text-[0.7rem] px-2.5 py-0.5 rounded-full border border-[var(--accent)] bg-[var(--accent-bg)] uppercase tracking-wider">
          {provider === 'github' ? 'GitHub AI' : 'KI-generiert'}
        </span>
      </div>

      <div className={`relative transition-all duration-400 ${isExpanded ? 'max-h-none' : 'max-h-[148px] overflow-hidden'}`}>
        <div 
          className="text-white/90 text-[0.95rem] leading-[1.8] summary-markdown"
          dangerouslySetInnerHTML={{ __html: formattedSummary }}
        />
        {!isExpanded && summary.length > 300 && (
          <div className="absolute bottom-0 left-0 right-0 h-14 bg-gradient-to-t from-[var(--summary-bg)] to-transparent pointer-events-none" />
        )}
      </div>

      {summary.length > 300 && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-2.5 flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--accent-bg)] border border-blue-500/40 text-[0.85rem] text-blue-100 hover:bg-blue-500/20 transition-all"
        >
          {isExpanded ? (
            <>Weniger anzeigen <ChevronUp className="w-3.5 h-3.5" /></>
          ) : (
            <>Mehr anzeigen <ChevronDown className="w-3.5 h-3.5" /></>
          )}
        </button>
      )}

      <div className="mt-4 pt-2.5 border-t border-white/10 text-[0.77rem] text-[var(--text-secondary)] italic flex items-center gap-2">
        <AlertTriangle className="w-3.5 h-3.5" /> Die KI kann Fehler machen. Überprüfe wichtige Informationen.
      </div>

      {!isError && sources.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {sources.map((source) => (
            <a
              key={source.index}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[var(--surface)] border border-[var(--border)] rounded-full px-2.5 py-1 text-[0.75rem] text-[var(--link)] hover:bg-[var(--surface-hover)] transition-colors truncate max-w-[260px]"
            >
              [{source.index}] {new URL(source.url).hostname}
            </a>
          ))}
        </div>
      )}

      {!isError && (
        <div className="mt-5 pt-4 border-t border-[var(--border)] flex items-center gap-3 text-[0.8rem] text-[var(--text-secondary)]">
          <span>War diese Antwort korrekt?</span>
          {!feedbackSent ? (
            <>
              <button
                onClick={() => handleFeedback(1)}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--border)] hover:border-green-500/50 hover:bg-green-500/10 hover:text-green-500 transition-all font-medium"
              >
                <ThumbsUp className="w-4 h-4" /> Ja
              </button>
              <button
                onClick={() => handleFeedback(-1)}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--border)] hover:border-red-500/50 hover:bg-red-500/10 hover:text-red-500 transition-all font-medium"
              >
                <ThumbsDown className="w-4 h-4" /> Nein
              </button>
            </>
          ) : (
            <span className="text-green-500 font-semibold animate-in fade-in slide-in-from-left-2">
              ✨ Danke! Dein Feedback hilft uns besser zu werden.
            </span>
          )}
        </div>
      )}

      {/* Model/Provider info is now at the top */}
    </div>
  );
}

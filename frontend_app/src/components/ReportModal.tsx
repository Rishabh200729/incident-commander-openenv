import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  reportContent: string | null;
}

export default function ReportModal({ isOpen, onClose, reportContent }: ReportModalProps) {
  if (!isOpen || !reportContent) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-300">
        <div className="flex items-center justify-between p-4 border-b border-outline-variant/10 bg-surface">
          <h2 className="text-lg font-h2 font-bold text-on-surface">Post-Incident Report</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-on-surface-variant hover:bg-outline-variant/20 hover:text-on-surface transition-colors"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto flex-1 prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({node, ...props}) => <div className="overflow-x-auto"><table className="min-w-full border-collapse" {...props} /></div>,
              th: ({node, ...props}) => <th className="border border-outline-variant/20 px-4 py-2 bg-surface text-left font-bold" {...props} />,
              td: ({node, ...props}) => <td className="border border-outline-variant/20 px-4 py-2" {...props} />,
            }}
          >
            {reportContent}
          </ReactMarkdown>
        </div>
        
        <div className="p-4 border-t border-outline-variant/10 bg-surface flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-primary/20 text-primary border border-primary/30 rounded font-semibold text-sm hover:bg-primary/30 transition-colors"
          >
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
}

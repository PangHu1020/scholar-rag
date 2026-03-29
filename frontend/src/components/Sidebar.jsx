import { useState } from 'react';
import { Plus, Trash2, MessageSquare } from 'lucide-react';

export default function Sidebar({ sessions, currentId, onSelect, onNew, onDelete }) {
  return (
    <div className="w-64 h-screen bg-gray-50 border-r border-gray-200 flex flex-col">
      <div className="p-4">
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-4 py-2.5 rounded-lg border border-gray-200 hover:bg-gray-100 text-sm text-gray-700 transition-colors"
        >
          <Plus size={16} />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2">
        {sessions.map((s) => (
          <div
            key={s.session_id}
            onClick={() => onSelect(s.session_id)}
            className={`group flex items-center gap-2 px-3 py-2.5 mx-1 mb-0.5 rounded-lg cursor-pointer text-sm transition-colors ${
              currentId === s.session_id
                ? 'bg-gray-200 text-gray-900'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <MessageSquare size={14} className="shrink-0 text-gray-400" />
            <span className="flex-1 truncate">
              {s.title || 'New Chat'}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.session_id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-300 transition-opacity"
            >
              <Trash2 size={13} className="text-gray-400" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

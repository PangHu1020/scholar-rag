import { useState, useEffect } from 'react';
import { Trash2, FileText, Database, Loader2 } from 'lucide-react';
import { fetchFiles, deleteFile, clearCollection } from '../api';

export default function SettingsPanel({ onCollectionCleared }) {
  const [files, setFiles] = useState([]);
  const [clearing, setClearing] = useState(false);

  const load = () => fetchFiles().then(setFiles).catch(() => {});
  useEffect(() => { load(); }, []);

  const handleDeleteFile = async (fileId) => {
    await deleteFile(fileId);
    load();
  };

  const handleClear = async () => {
    if (!confirm('This will delete ALL data from the vector database. Continue?')) return;
    setClearing(true);
    try {
      await clearCollection();
      setFiles([]);
      if (onCollectionCleared) onCollectionCleared();
    } catch {}
    setClearing(false);
  };

  return (
    <div className="p-4 space-y-4">
      <div>
        <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Files</h3>
        {files.length === 0 ? (
          <p className="text-xs text-gray-400">No files uploaded</p>
        ) : (
          <div className="space-y-1">
            {files.map((f) => (
              <div key={f.file_id} className="flex items-center gap-2 text-xs text-gray-600 px-2 py-1.5 rounded hover:bg-gray-50">
                <FileText size={13} className="text-gray-400 shrink-0" />
                <span className="truncate flex-1">{f.filename}</span>
                <span className="text-gray-400 shrink-0">{f.chunk_count}c</span>
                <button onClick={() => handleDeleteFile(f.file_id)} className="p-0.5 rounded hover:bg-gray-200">
                  <Trash2 size={12} className="text-gray-400" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-gray-100 pt-4">
        <button
          onClick={handleClear}
          disabled={clearing}
          className="flex items-center gap-2 text-xs text-red-600 hover:text-red-700 disabled:text-gray-400"
        >
          {clearing ? <Loader2 size={13} className="animate-spin" /> : <Database size={13} />}
          Clear Database
        </button>
      </div>
    </div>
  );
}

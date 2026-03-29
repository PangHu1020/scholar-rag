import { useState, useRef } from 'react';
import { Upload, X, FileText, Loader2 } from 'lucide-react';
import { uploadFiles } from '../api';

export default function FileUpload({ onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState([]);
  const inputRef = useRef(null);

  const handleFiles = async (fileList) => {
    const pdfs = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length === 0) return;

    setUploading(true);
    setResults([]);
    try {
      const res = await uploadFiles(pdfs);
      setResults(res.files || []);
      if (onUploaded) onUploaded();
    } catch (err) {
      setResults([{ filename: 'Upload failed', status: 'error', detail: err.message }]);
    }
    setUploading(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="p-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          dragging ? 'border-black bg-gray-50' : 'border-gray-200 hover:border-gray-300'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {uploading ? (
          <Loader2 size={24} className="mx-auto text-gray-400 animate-spin" />
        ) : (
          <Upload size={24} className="mx-auto text-gray-400" />
        )}
        <p className="text-sm text-gray-500 mt-2">
          {uploading ? 'Processing...' : 'Drop PDF files here or click to browse'}
        </p>
      </div>

      {results.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {results.map((r, i) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${
              r.status === 'ok' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
            }`}>
              <FileText size={13} />
              <span className="truncate flex-1">{r.filename}</span>
              {r.status === 'ok' ? (
                <span>{r.chunk_count} chunks</span>
              ) : (
                <span>{r.detail}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

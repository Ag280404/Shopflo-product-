"use client";

import { useState } from "react";

type ExportButtonProps = {
  filename: string;
  data: unknown;
};

const ExportButton = ({ filename, data }: ExportButtonProps) => {
  const [downloaded, setDownloaded] = useState(false);
  const handleExport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2000);
  };

  return (
    <button
      type="button"
      onClick={handleExport}
      className="rounded bg-slate-900 px-4 py-2 text-xs font-semibold text-white"
    >
      {downloaded ? "Exported" : "Export experiment plan"}
    </button>
  );
};

export default ExportButton;

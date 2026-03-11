"use client";

import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { downloadCSV } from "@/lib/utils";
import { cn } from "@/lib/utils";

interface ExportButtonProps {
  data: Record<string, unknown>[];
  filename: string;
  columns?: { key: string; header: string }[];
  className?: string;
}

export function ExportButton({ data, filename, columns, className }: ExportButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      disabled={!data || data.length === 0}
      className={cn("gap-1.5", className)}
      onClick={() => downloadCSV(data, filename, columns)}
    >
      <Download className="size-3.5" />
      Export CSV
    </Button>
  );
}

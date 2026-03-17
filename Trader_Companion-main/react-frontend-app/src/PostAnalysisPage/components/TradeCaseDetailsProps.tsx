import React, { useCallback, useEffect, useRef, useState, forwardRef } from "react";
import { Trade } from "@/TradeHistoryPage/types/Trade";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { analysisService, PostTradeAnalysis } from "../services/postAnalysis";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";

interface TradeCaseDetailsProps {
  trade: Trade;
  onAnalysesChanged?: (tradeId: number, analyses: PostTradeAnalysis[]) => void;
  onRequestFullscreen?: (imageIndex: number) => void; // index within current images list
}

// forwardRef so parent can focus the drop zone when navigating trades via keyboard
const TradeCaseDetails = forwardRef<HTMLDivElement, TradeCaseDetailsProps>(({ trade, onAnalysesChanged, onRequestFullscreen }, ref) => {
  // Hooks must be first
  const [existingAnalyses, setExistingAnalyses] = useState<PostTradeAnalysis[]>([]);
  // Trade-level description (not per image)
  const [description, setDescription] = useState("");
  const [descriptionAnalysisId, setDescriptionAnalysisId] = useState<number | null>(null);
  // Form state for creating a NEW image entry (image only)
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  type CaseData = {
    symbol?: string;
    total_score?: number;
    personal_opinion_score?: number;
    details?: string;
    demand_reason?: string;
    characteristics?: Record<string, unknown>;
  };
  let caseData: CaseData | null = null;
  if (trade.Case) {
    try {
      caseData = typeof trade.Case === "string" ? (JSON.parse(trade.Case) as CaseData) : (trade.Case as unknown as CaseData);
    } catch (err) {
      console.error("Invalid Case JSON:", err);
    }
  }

  const { symbol, total_score, personal_opinion_score, details, demand_reason, characteristics = {} } =
    caseData || {};
  // Get Trade Note (Exit_Reason) from trade
  const tradeNote = trade.Exit_Reason;

  const loadAnalyses = useCallback(async () => {
    try {
      const data = await analysisService.listByTrade(trade.ID);
      const sorted = [...data].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
      // Identify description record: first with notes and (no image OR treat first notes as description)
      const desc = sorted.find(a => (a.notes && a.notes.trim().length > 0));
      if (desc) {
        setDescription(desc.notes || "");
        setDescriptionAnalysisId(desc.image ? null : desc.id); // Only bind id if it's a pure description (no image)
      } else {
        setDescription("");
        setDescriptionAnalysisId(null);
      }
      setExistingAnalyses(sorted);
      onAnalysesChanged?.(trade.ID, sorted);
    } catch (e) {
      console.error(e);
    }
  }, [trade.ID, onAnalysesChanged]);

  useEffect(() => {
    loadAnalyses();
  }, [loadAnalyses]);

  const handleFiles = (files: FileList | null) => {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      setError("Only image files are allowed");
      return;
    }
    setError(null);
    setImageFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    handleFiles(e.dataTransfer.files);
  };
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const submitNewImage = async () => {
    if (!imageFile) { setError("Choose an image first"); return; }
    setIsSubmitting(true); setError(null);
    try { await analysisService.create({ trade_id: trade.ID, imageFile }); setImageFile(null); await loadAnalyses(); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed to save image"); }
    finally { setIsSubmitting(false); }
  };

  const saveDescription = async () => {
    setIsSubmitting(true); setError(null);
    try {
      if (descriptionAnalysisId) {
        await analysisService.update(descriptionAnalysisId, { notes: description });
      } else {
        // Create a pure description entry (no image)
        const created = await analysisService.create({ trade_id: trade.ID, notes: description });
        // If created without image and has notes, remember its id for updates
        if (!created.image) setDescriptionAnalysisId(created.id);
      }
      await loadAnalyses();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save description");
    } finally { setIsSubmitting(false); }
  };

  const deleteAnalysis = async (id: number) => {
    try {
      await analysisService.delete(id);
      setExistingAnalyses(prev => {
        const next = prev.filter(a => a.id !== id);
        if (descriptionAnalysisId === id) { setDescriptionAnalysisId(null); }
        onAnalysesChanged?.(trade.ID, next);
        return next;
      });
    } catch (e) {
      console.error(e);
      setError("Failed to delete image");
    }
  };

  // Determine whether we have meaningful case data (avoid blocking UI if invalid/missing)
  const showCaseDetails = Boolean(
    caseData && (
      caseData.symbol ||
      caseData.total_score !== undefined ||
      caseData.personal_opinion_score !== undefined ||
      caseData.details ||
      caseData.demand_reason ||
      (caseData.characteristics && Object.keys(caseData.characteristics || {}).length > 0)
    )
  );

  // Preview for newly selected (unsaved) image
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreviewUrl(null);
  }, [imageFile]);

  return (
    <Card className="mt-2 border border-border shadow-md rounded-xl">
      <CardContent className="p-4 space-y-6">
        <div className="space-y-6">
          <h4 className="font-semibold text-md">Post Analysis</h4>

          {/* Trade-level description */}
          <div className="space-y-2">
            <Textarea
              placeholder="Overall trade description / observations..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="min-h-[140px]"
            />
            <div className="flex gap-2">
              <Button size="sm" disabled={isSubmitting} onClick={saveDescription}>
                {isSubmitting ? 'Saving...' : 'Save Description'}
              </Button>
              {description && descriptionAnalysisId && (
                <Button size="sm" variant="secondary" disabled={isSubmitting} onClick={() => { setDescription(''); setDescriptionAnalysisId(null); }}>
                  Clear (not saved)
                </Button>
              )}
            </div>
          </div>

          {/* Existing images list (before add form so centering finds first image) */}
          <div className="space-y-6" ref={ref} tabIndex={-1}>
            {existingAnalyses.filter(a => a.image).length === 0 && <p className="text-sm text-muted-foreground">No images yet.</p>}
            {existingAnalyses.filter(a => a.image).map((analysis, idx) => (
              <div key={analysis.id} className="space-y-2">
                <div className="relative group">
                  <img
                    src={analysis.image}
                    alt={analysis.title || 'analysis'}
                    data-analysis-image
                    className="w-full h-auto max-h-[65vh] object-contain rounded border border-border cursor-zoom-in"
                    onClick={() => onRequestFullscreen?.(idx)}
                    title="Click to open fullscreen"
                  />
                  <button
                    type="button"
                    onClick={() => deleteAnalysis(analysis.id)}
                    className="absolute top-2 right-2 bg-red-600/80 hover:bg-red-600 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition"
                  >Delete</button>
                  <button
                    type="button"
                    onClick={() => onRequestFullscreen?.(idx)}
                    className="absolute bottom-2 right-2 bg-black/60 hover:bg-black/70 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition"
                  >Fullscreen</button>
                </div>
                <hr className="border-border" />
              </div>
            ))}
          </div>

          {/* Add new image form */}
          <div
            className="border-2 border-dashed rounded-md p-3 w-full text-center cursor-pointer hover:bg-muted/40 transition focus:outline-none focus:ring-2 focus:ring-ring/60"
            aria-label="New analysis image drop zone"
            onDrop={onDrop}
            onDragOver={onDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <div className="space-y-2 relative">
                <img src={previewUrl} alt="preview" className="w-full h-auto max-h-[60vh] object-contain rounded" />
                <p className="text-xs text-muted-foreground">(Unsaved) {imageFile?.name} — click or drop to replace</p>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Drop new image here or click to browse</p>
            )}
            <Input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={isSubmitting || !imageFile} onClick={submitNewImage}>
              {isSubmitting ? 'Saving...' : 'Add Image'}
            </Button>
            {imageFile && (
              <Button size="sm" variant="secondary" onClick={() => setImageFile(null)}>Clear Image</Button>
            )}
          </div>

          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
        {showCaseDetails && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Case Details</h3>
            {/* Case JSON details if present */}
            {showCaseDetails && (
              <div className="grid grid-cols-2 gap-0 text-sm">
                <p className="border-r border-b border-border" style={{ margin: 0, padding: '2px 6px' }}><span className="font-medium">Symbol:</span> {symbol}</p>
                <p className="border-b border-border" style={{ margin: 0, padding: '2px 6px' }}><span className="font-medium">Total Score:</span> {total_score}</p>
                <p className="border-r border-b border-border" style={{ margin: 0, padding: '2px 6px' }}><span className="font-medium">Opinion Score:</span> {personal_opinion_score}</p>
                <p className="border-b border-border" style={{ margin: 0, padding: '2px 6px' }}><span className="font-medium">Demand Reason:</span> {demand_reason || '—'}</p>
              </div>
            )}
            {tradeNote && (
              <div className="grid grid-cols-1 gap-0 text-sm">
                <p className="border-b border-border" style={{ margin: 0, padding: '2px 6px' }}><span className="font-medium">Trade Note:</span> {tradeNote}</p>
              </div>
            )}
            {details && (
              <div>
                <p className="font-medium">Details:</p>
                <p className="text-muted-foreground text-sm">{details}</p>
              </div>
            )}
            {showCaseDetails && (
              <div>
                <p className="font-medium mb-2">Characteristics:</p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-0 text-sm">
                  {Object.entries(characteristics).map(([key, value], idx) => (
                    <div
                      key={key}
                      className={`flex items-center space-x-2 border-b border-border ${((idx + 1) % 3 !== 0) ? 'border-r' : ''}`}
                      style={{ padding: '2px 6px', margin: 0 }}
                    >
                      <Checkbox checked={Boolean(value)} disabled />
                      <span>{key}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <hr className="border-border" />
          </div>
        )}

      </CardContent>
    </Card>
  );
});

TradeCaseDetails.displayName = "TradeCaseDetails";

export default TradeCaseDetails;

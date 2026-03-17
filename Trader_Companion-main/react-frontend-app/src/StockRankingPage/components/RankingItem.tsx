import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { StockPick, GlobalCharacteristic } from '../types';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, ChevronUp, X, Edit, Check, MoreHorizontal, Download, AlertTriangle } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Checkbox
} from "@/components/ui/checkbox";
import { globalCharacteristicsApi } from '../services/globalCharacteristics';
import { stockPicksApi } from '../services/stockPick';
import { tradeAPI } from '@/TradeStatisticsPage/services/tradeAPI';
import type { Trade } from '@/TradeHistoryPage/types/Trade';

interface Props {
  stock: StockPick;
  onUpdate: (updatedStock: StockPick) => void;
  onRemove: () => void;
}

type ManualSaveField = 'note' | 'catalyst' | 'details';

export const RankingItem: React.FC<Props> = ({
  stock: initialStock,
  onUpdate,
  onRemove
}) => {
  const [stock, setStock] = useState<StockPick>(initialStock);
  const [isExpanded, setIsExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailsText, setCaseText] = useState(initialStock.case_text || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [globalCharacteristics, setGlobalCharacteristics] = useState<GlobalCharacteristic[]>([]);
  // Meta lists from backend (ordered, priority, color-coded)
  const [orderedMeta, setOrderedMeta] = useState<{ id: number; characteristic_id: number; name: string; position: number; }[]>([]);
  const [priorityMeta, setPriorityMeta] = useState<{ id: number; characteristic_id: number; name: string; }[]>([]);
  const [colorCodedMeta, setColorCodedMeta] = useState<{ id: number; characteristic_id: number; name: string; }[]>([]);
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [isEditingScore, setIsEditingScore] = useState<number | null>(null);
  const [editedScore, setEditedScore] = useState<string>('0');
  const [isEditingPersonalScore, setIsEditingPersonalScore] = useState(false);
  const [personalScore, setPersonalScore] = useState<string>(String(initialStock.personal_opinion_score || 0));
  // Properly type the visible characteristics array
  const [visibleCharacteristics, setVisibleCharacteristics] = useState<StockPick['characteristics']>([]);
  const [priorityCharacteristics, setPriorityCharacteristics] = useState<StockPick['characteristics']>([]);
  const [hasHiddenCharacteristics, setHasHiddenCharacteristics] = useState(false);
  const [note, setNote] = useState<string>(initialStock.note || '');
  const [pendingCharacteristics, setPendingCharacteristics] = useState<Record<number, boolean>>({});
  const [lastClickedCharacteristic, setLastClickedCharacteristic] = useState<number | null>(null);
  const [hasTradeInHistory, setHasTradeInHistory] = useState<boolean | null>(null);
  const [isCheckingTrades, setIsCheckingTrades] = useState(false);
  const [manualSavingField, setManualSavingField] = useState<ManualSaveField | null>(null);

  // Track if user is actively editing fields to avoid overwriting from server
  const [isEditingCaseText, setIsEditingCaseText] = useState(false);
  const [isEditingNote, setIsEditingNote] = useState(false);
  const [isEditingDemandReason, setIsEditingDemandReason] = useState(false);

  // Version counter to ignore stale save responses
  const requestVersionRef = useRef(0);
  // saving indicator removed
  const hasSyncedLocalOverridesRef = useRef(false);

  // Local storage key helper (prefer id, fallback to symbol)
  const storageKey = `rankingItem:${initialStock.id || initialStock.symbol}`;

  // Persist important front-end fields immediately for resilience on refresh
  const persistLocal = useCallback((data: Partial<StockPick>) => {
    try {
      const existing = JSON.parse(localStorage.getItem(storageKey) || '{}');
      const merged = { ...existing, ...data };
      localStorage.setItem(storageKey, JSON.stringify(merged));
    } catch {
      // ignore storage errors
    }
  }, [storageKey]);


  // Track pending characteristic changes
  // saving indicator removed

  // Add a saveTimeout ref to debounce saving
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Ref for the characteristics container
  const charContainerRef = useRef<HTMLDivElement>(null);
  // Header responsive shrinking refs
  const headerRef = useRef<HTMLDivElement>(null);
  const regularCharsWrapperRef = useRef<HTMLSpanElement>(null); // regular characteristics (common)
  const priorityCharsWrapperRef = useRef<HTMLSpanElement>(null); // yellow priority
  const noteWrapperRef = useRef<HTMLSpanElement>(null);
  const catalystWrapperRef = useRef<HTMLSpanElement>(null);
  const hasFetchedMetaRef = useRef(false);

  // Track which groups are hidden (for className binding only)
  const [hideRegularChars, setHideRegularChars] = useState(false);
  const [hidePriorityChars, setHidePriorityChars] = useState(false);
  const [hideNote, setHideNote] = useState(false);
  const [hideCatalyst, setHideCatalyst] = useState(false);

  // Trade existence checker (memoized)
  const checkTradeExists = useCallback(async () => {
    try {
      setIsCheckingTrades(true);
      const resp = await tradeAPI.getTrades();
      const trades: Trade[] = resp.data;
      const exists = trades.some(t => t.Ticker.toUpperCase() === stock.symbol.toUpperCase());
      setHasTradeInHistory(exists);
    } catch {
      setHasTradeInHistory(false);
    } finally {
      setIsCheckingTrades(false);
    }
  }, [stock.symbol]);

  // Load local overrides on first mount & whenever initialStock id changes
  useEffect(() => {
    // Load overrides
    let overrides: Partial<StockPick> = {};
    try {
      overrides = JSON.parse(localStorage.getItem(storageKey) || '{}');
    } catch { /* ignore */ }

    // Merge server -> local with local authoritative for TEXT fields only
    // personal_opinion_score always comes from server (authoritative source)
    const merged: StockPick = {
      ...initialStock,
      case_text: overrides.case_text ?? initialStock.case_text,
      note: overrides.note ?? initialStock.note,
      demand_reason: overrides.demand_reason ?? initialStock.demand_reason,
      // personal_opinion_score is NOT overridden from localStorage - server is authoritative
    } as StockPick;
    setStock(merged);
    setCaseText(merged.case_text || '');
    setPersonalScore(String(initialStock.personal_opinion_score || 0));
    setNote(merged.note || '');

    // Persist text fields only (personal_opinion_score not cached - server is authoritative)
    persistLocal({
      case_text: merged.case_text,
      note: merged.note,
      demand_reason: merged.demand_reason
    });

    // After initial merge, if overrides differ from backend, push update once
    if (!hasSyncedLocalOverridesRef.current) {
      const diff = (
        overrides.case_text !== undefined && overrides.case_text !== initialStock.case_text) ||
        (overrides.note !== undefined && overrides.note !== initialStock.note) ||
        (overrides.demand_reason !== undefined && overrides.demand_reason !== initialStock.demand_reason);
      if (diff) {
        (async () => {
          try {
            await stockPicksApi.updateStockPick(initialStock.id, {
              case_text: merged.case_text,
              note: merged.note,
              demand_reason: merged.demand_reason,
              ranking_box: merged.ranking_box,
              symbol: merged.symbol,
              total_score: merged.total_score,
              personal_opinion_score: merged.personal_opinion_score
            });
          } catch {/* ignore sync error */ }
        })();
      }
      hasSyncedLocalOverridesRef.current = true;
    }

    const priorityNamesInit = new Set(priorityMeta.map(p => p.name));
    const priorityChars = merged.characteristics.filter(char => priorityNamesInit.has(char.name));
    setPriorityCharacteristics(priorityChars);
    if (isExpanded && hasTradeInHistory === null) {
      checkTradeExists();
    }
    // We intentionally exclude certain stable callbacks to avoid needless re-sync loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialStock.id, storageKey, priorityMeta]);

  // When parent passes a changed initialStock (other than id), merge but don't override active edits
  useEffect(() => {
    setStock(prev => {
      // If ids differ we've already handled via previous effect
      if (prev.id !== initialStock.id) return prev;
      const merged: StockPick = {
        ...prev,
        ...initialStock,
        case_text: isEditingCaseText ? prev.case_text : prev.case_text || initialStock.case_text,
        note: isEditingNote ? prev.note : prev.note || initialStock.note,
        demand_reason: isEditingDemandReason ? prev.demand_reason : prev.demand_reason || initialStock.demand_reason,
        personal_opinion_score: isEditingPersonalScore ? prev.personal_opinion_score : initialStock.personal_opinion_score,
      } as StockPick;
      // Update personalScore state when not editing
      if (!isEditingPersonalScore) {
        setPersonalScore(String(initialStock.personal_opinion_score || 0));
      }
      return merged;
    });
  }, [initialStock, isEditingCaseText, isEditingNote, isEditingDemandReason, isEditingPersonalScore]);

  // Calculate which characteristics should be visible based on container height
  // Use useCallback to memoize the function and prevent unnecessary re-renders
  const calculateVisibleCharacteristics = useCallback(() => {
    if (!charContainerRef.current) return;

    // Get all characteristics and sort by score (descending)
    // Filter out priority characteristics as they'll be shown separately
    const priorityCharNames = priorityMeta
      .map(p => p.name)
      .filter(name => stock.characteristics.some(c => c.name === name));

    const sortedChars = [...stock.characteristics]
      .filter(char => !priorityCharNames.includes(char.name))
      .sort((a, b) => b.score - a.score);

    const container = charContainerRef.current;
    // Clone the container for testing
    const testContainer = container.cloneNode(true) as HTMLDivElement;
    testContainer.style.position = 'absolute';
    testContainer.style.visibility = 'hidden';
    testContainer.style.width = `${container.offsetWidth}px`;
    testContainer.innerHTML = '';
    document.body.appendChild(testContainer);

    // First add all priority characteristic badges
    const priorityBadges = document.createElement('div');
    priorityBadges.style.display = 'inline-block';
    testContainer.appendChild(priorityBadges);

    // Calculate how many badges we can fit before creating more than 2 rows
    const maxHeight = 28; // Approximate height for 1 row (adjust as needed)
    let visibleCount = 0;
    const totalChars = sortedChars.length;

    // If we have no non-priority characteristics, early return
    if (totalChars === 0) {
      setVisibleCharacteristics([]);
      setHasHiddenCharacteristics(false);
      document.body.removeChild(testContainer);
      return;
    }

    for (let i = 0; i < totalChars; i++) {
      const badge = document.createElement('div');
      badge.className = 'inline-block px-2 py-0.5 m-0.5 text-xs font-medium rounded-full';
      badge.textContent = sortedChars[i].name;
      testContainer.appendChild(badge);

      if (testContainer.offsetHeight > maxHeight) {
        // We've gone too far, remove this badge
        testContainer.removeChild(badge);

        // If we have more chars and this isn't the first one
        if (i < totalChars - 1 && i > 0) {
          visibleCount = i;
          setHasHiddenCharacteristics(true);
        } else {
          visibleCount = Math.max(1, i);
          setHasHiddenCharacteristics(i < totalChars - 1);
        }
        break;
      }

      // If we've added all badges and still fit
      if (i === totalChars - 1) {
        visibleCount = totalChars;
        setHasHiddenCharacteristics(false);
      }
    }

    // Clean up
    document.body.removeChild(testContainer);

    // If we need to add the "more" indicator, reserve space for it
    if (visibleCount < totalChars) {
      const moreTestContainer = container.cloneNode(true) as HTMLDivElement;
      moreTestContainer.style.position = 'absolute';
      moreTestContainer.style.visibility = 'hidden';
      moreTestContainer.style.width = `${container.offsetWidth}px`;
      moreTestContainer.innerHTML = '';
      document.body.appendChild(moreTestContainer);

      // Add the priority badges first
      const priorityBadges = document.createElement('div');
      priorityBadges.style.display = 'inline-block';
      moreTestContainer.appendChild(priorityBadges);

      // Add visible badges
      for (let i = 0; i < visibleCount; i++) {
        const badge = document.createElement('div');
        badge.className = 'inline-block px-2 py-0.5 m-0.5 text-xs font-medium rounded-full';
        badge.textContent = sortedChars[i].name;
        moreTestContainer.appendChild(badge);
      }

      // Add the "more" badge
      const moreBadge = document.createElement('div');
      moreBadge.className = 'inline-block px-2 py-0.5 m-0.5 text-xs font-medium rounded-full';
      moreBadge.textContent = '...';
      moreTestContainer.appendChild(moreBadge);

      // Check if adding "more" badge pushes to third row
      if (moreTestContainer.offsetHeight > maxHeight && visibleCount > 1) {
        visibleCount--;
      }

      document.body.removeChild(moreTestContainer);
    }

    // Update state with visible characteristics
    setVisibleCharacteristics(sortedChars.slice(0, visibleCount));
    setHasHiddenCharacteristics(visibleCount < totalChars);
  }, [stock.characteristics, priorityMeta]);

  // Fetch meta lists (defined before effect usage)
  const fetchMetaLists = useCallback(async () => {
    setMetaLoading(true); setMetaError(null);
    try {
      const module = await import('../services/characteristicMeta');
      const [orderedRes, priorityRes, colorRes] = await Promise.all([
        module.characteristicMetaApi.getOrdered(),
        module.characteristicMetaApi.getPriority(),
        module.characteristicMetaApi.getColorCoded()
      ]);
      setOrderedMeta(orderedRes.data as { id: number; characteristic_id: number; name: string; position: number; }[]);
      setPriorityMeta(priorityRes.data as { id: number; characteristic_id: number; name: string; }[]);
      setColorCodedMeta(colorRes.data as { id: number; characteristic_id: number; name: string; }[]);
    } catch (e) {
      console.error('Error fetching characteristic meta lists', e);
      setMetaError('Failed loading characteristic meta lists');
      throw e;
    } finally { setMetaLoading(false); }
  }, []);

  const ensureMetaListsLoaded = useCallback(async () => {
    if (hasFetchedMetaRef.current) return;
    try {
      await fetchMetaLists();
      hasFetchedMetaRef.current = true;
    } catch {
      // retry next time if fetch fails
    }
  }, [fetchMetaLists]);

  // Fetch global & meta characteristics when component expands
  useEffect(() => {
    if (isExpanded) {
      fetchGlobalCharacteristics();
      ensureMetaListsLoaded();
      if (hasTradeInHistory === null) {
        checkTradeExists();
      }
    }
  }, [isExpanded, hasTradeInHistory, checkTradeExists, ensureMetaListsLoaded]);

  useEffect(() => {
    ensureMetaListsLoaded();
  }, [ensureMetaListsLoaded]);

  // Check visible characteristics on stock change or window resize
  useEffect(() => {
    const dynPriorityNames = new Set(priorityMeta.map(p => p.name));
    const priorityChars = stock.characteristics.filter(char => dynPriorityNames.has(char.name));
    setPriorityCharacteristics(priorityChars);

    // Then calculate remaining visible characteristics
    calculateVisibleCharacteristics();
    window.addEventListener('resize', calculateVisibleCharacteristics);
    return () => {
      window.removeEventListener('resize', calculateVisibleCharacteristics);
    };
  }, [calculateVisibleCharacteristics, stock.characteristics, priorityMeta]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  // Progressive hiding logic: hide from right to left -> regular characteristics, priority (yellow), note, catalyst
  useEffect(() => {
    const headerEl = headerRef.current;
    if (!headerEl) return;

    // Function that applies hiding logic
    const applySizing = () => {
      const groups: Array<{ ref: React.RefObject<HTMLElement>, hideSetter: (v: boolean) => void }> = [
        { ref: regularCharsWrapperRef, hideSetter: setHideRegularChars },
        { ref: priorityCharsWrapperRef, hideSetter: setHidePriorityChars },
        { ref: noteWrapperRef, hideSetter: setHideNote },
        { ref: catalystWrapperRef, hideSetter: setHideCatalyst }
      ];

      // First show everything
      groups.forEach(g => {
        if (g.ref.current) g.ref.current.style.display = '';
        g.hideSetter(false);
      });

      // Allow layout to update before measuring
      requestAnimationFrame(() => {
        // Hide progressively while overflowing
        for (const g of groups) {
          if (headerEl.scrollWidth <= headerEl.clientWidth) break;
          if (g.ref.current) {
            g.ref.current.style.display = 'none';
            g.hideSetter(true);
          }
        }
      });
    };

    // Resize observer to react to width changes
    const ro = new ResizeObserver(() => applySizing());
    ro.observe(headerEl);

    // Also apply when content changes (characteristics, note, catalyst)
    applySizing();
    return () => {
      ro.disconnect();
    };
  }, [visibleCharacteristics, priorityCharacteristics, note, stock.demand_reason, stock.personal_opinion_score, stock.total_score]);

  // Fetch global characteristics from API
  const fetchGlobalCharacteristics = async () => {
    try {
      const response = await globalCharacteristicsApi.getAllGlobalCharacteristics();
      setGlobalCharacteristics(response.data);
    } catch (err) {
      console.error('Error fetching global characteristics:', err);
      setError('Failed to load global characteristics');
    }
  };
  // (fetchMetaLists definition moved earlier)

  // checkTradeExists now defined via useCallback above

  // Handle case text changes - use debounce to prevent API calls on every keystroke
  // Update the handleDetailsTextChange function to use the latest stock state
  const handleDetailsTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newCaseText = e.target.value;
    setCaseText(newCaseText);
    setIsEditingCaseText(true);
    setStock(prevStock => ({ ...prevStock, case_text: newCaseText }));
    persistLocal({ case_text: newCaseText });

    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    const currentRequestVersion = ++requestVersionRef.current;
    saveTimeoutRef.current = setTimeout(async () => {
      try {
        // saving indicator removed
        const currentStock = { ...stock, case_text: newCaseText };
        await stockPicksApi.updateStockPick(currentStock.id, {
          case_text: newCaseText,
          ranking_box: currentStock.ranking_box,
          symbol: currentStock.symbol,
          total_score: currentStock.total_score,
          personal_opinion_score: currentStock.personal_opinion_score,
          note: currentStock.note,
          demand_reason: currentStock.demand_reason
        }).then(resp => {
          // Ignore stale responses
          if (currentRequestVersion !== requestVersionRef.current) return;
          // Merge without overwriting if user kept typing (already same value)
          setStock(prev => ({ ...prev, ...resp.data, case_text: prev.case_text }));
          onUpdate({ ...resp.data, case_text: newCaseText });
        });
      } catch (err) {
        console.error('Error saving case text:', err);
        setError('Failed to save case text');
      } finally {
        // saving indicator removed
        setIsEditingCaseText(false); // user paused typing
      }
    }, 900); // faster debounce while ensuring stability
  };

  const manualSaveField = useCallback(async (field: ManualSaveField, value: string) => {
    if (!stock.id) return;
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }

    setManualSavingField(field);
    const currentRequestVersion = ++requestVersionRef.current;

    const payload = {
      case_text: field === 'details' ? value : detailsText,
      note: field === 'note' ? value : note,
      demand_reason: field === 'catalyst' ? value : (stock.demand_reason ?? ''),
      ranking_box: stock.ranking_box,
      symbol: stock.symbol,
      total_score: stock.total_score,
      personal_opinion_score: stock.personal_opinion_score
    };

    try {
      const response = await stockPicksApi.updateStockPick(stock.id, payload);
      if (currentRequestVersion === requestVersionRef.current) {
        let nextStock: StockPick | null = null;
        setStock(prev => {
          const merged = {
            ...prev,
            ...response.data,
            case_text: field === 'details' ? value : prev.case_text,
            note: field === 'note' ? value : prev.note,
            demand_reason: field === 'catalyst' ? value : prev.demand_reason,
          } as StockPick;
          nextStock = merged;
          return merged;
        });

        if (field === 'note') {
          setNote(value);
        } else if (field === 'details') {
          setCaseText(value);
        }

        persistLocal({
          case_text: payload.case_text,
          note: payload.note,
          demand_reason: payload.demand_reason
        });

        if (nextStock) {
          onUpdate(nextStock);
        }
      }
    } catch (err) {
      console.error('Error saving field manually:', err);
      setError('Failed to save changes');
    } finally {
      setManualSavingField(null);
    }
  }, [detailsText, note, onUpdate, persistLocal, stock]);

  const handleManualSaveShortcut = useCallback((
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
    field: ManualSaveField
  ) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      event.stopPropagation();
      const target = event.target as (HTMLInputElement | HTMLTextAreaElement);
      manualSaveField(field, target.value);
    }
  }, [manualSaveField]);

  // Toggle a characteristic on/off
  const handleToggleCharacteristic = async (characteristicId: number, checked: boolean) => {
    try {
      // Immediately update UI state for better UX
      setPendingCharacteristics((prev: Record<number, boolean>) => ({ ...prev, [characteristicId]: checked }));
      setIsSubmitting(true);
      setError(null);

      // Optimistically update local state for immediate feedback
      if (checked) {
        // Find the global characteristic to get its default score
        const globalChar = globalCharacteristics.find(gc => gc.id === characteristicId);
        if (globalChar) {
          const optimisticChar = {
            id: -Date.now(), // Temporary negative ID to avoid conflicts
            name: globalChar.name,
            score: globalChar.default_score,
            characteristic_id: globalChar.id
          };

          // Add to local state immediately
          const newTotal = stock.total_score + globalChar.default_score;
          const updatedStock = {
            ...stock,
            characteristics: [...stock.characteristics, optimisticChar],
            total_score: newTotal
          };
          setStock(updatedStock);
          onUpdate(updatedStock);
        }
      } else {
        // Remove from local state immediately
        const charToRemove = stock.characteristics.find(c => c.characteristic_id === characteristicId);
        if (charToRemove) {
          const newTotal = stock.total_score - charToRemove.score;
          const updatedStock = {
            ...stock,
            characteristics: stock.characteristics.filter(c => c.characteristic_id !== characteristicId),
            total_score: newTotal
          };
          setStock(updatedStock);
          onUpdate(updatedStock);
        }
      }

      // Then make the actual API call
      let response;
      if (checked) {
        // Find the global characteristic to get its default score
        const globalChar = globalCharacteristics.find(gc => gc.id === characteristicId);
        const defaultScore = globalChar ? globalChar.default_score : 0;

        // Add the characteristic
        response = await stockPicksApi.addCharacteristic(stock.id, {
          characteristic_id: characteristicId,
          score: defaultScore
        });
      } else {
        // Remove the characteristic
        response = await stockPicksApi.removeCharacteristic(stock.id, {
          characteristic_id: characteristicId
        });
      }

      // Update with the actual server response
      setStock(response.data);
      onUpdate(response.data);
    } catch (err) {
      console.error('Error toggling characteristic:', err);
      setError(`Failed to ${checked ? 'add' : 'remove'} characteristic`);

      // Revert optimistic update on error
      setPendingCharacteristics((prev: Record<number, boolean>) => ({ ...prev, [characteristicId]: !checked }));
      const response = await stockPicksApi.getStockPick(stock.id);
      setStock(response.data);
      onUpdate(response.data);
    } finally {
      setIsSubmitting(false);
      setPendingCharacteristics((prev: Record<number, boolean>) => {
        const newState: Record<number, boolean> = { ...prev };
        delete newState[characteristicId];
        return newState;
      });
    }
  };

  // Begin editing a score
  const handleStartEditScore = (characteristicId: number) => {
    const char = stock.characteristics.find(c => c.characteristic_id === characteristicId);
    if (char) {
      setIsEditingScore(characteristicId);
      setEditedScore(String(char.score));
    }
  };

  // Save an edited score
  const handleSaveScore = async (characteristicId: number) => {
    try {
      setIsSubmitting(true);
      setError(null);

      // Parse the edited score string to number
      const numericScore = Number(editedScore) || 0;

      // Optimistically update the score
      const charToUpdate = stock.characteristics.find(c => c.characteristic_id === characteristicId);
      if (charToUpdate) {
        const scoreDiff = numericScore - charToUpdate.score;
        const updatedStock = {
          ...stock,
          characteristics: stock.characteristics.map(c =>
            c.characteristic_id === characteristicId
              ? { ...c, score: numericScore }
              : c
          ),
          total_score: stock.total_score + scoreDiff
        };
        setStock(updatedStock);
        onUpdate(updatedStock);
      }

      const response = await stockPicksApi.addCharacteristic(stock.id, {
        characteristic_id: characteristicId,
        score: numericScore
      });

      // Update with the actual server response
      setStock(response.data);
      onUpdate(response.data);
      setIsEditingScore(null);
    } catch (err) {
      console.error('Error updating score:', err);
      setError('Failed to update score');

      // Revert optimistic update on error
      const response = await stockPicksApi.getStockPick(stock.id);
      setStock(response.data);
      onUpdate(response.data);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle personal opinion score updates
  const handleStartEditPersonalScore = () => {
    setIsEditingPersonalScore(true);
  };

  const handleSavePersonalScore = async () => {
    try {
      setIsSubmitting(true);
      setError(null);

      // Parse the personal score string to number
      const numericPersonalScore = Number(personalScore) || 0;

      // Optimistically update the score
      const scoreDiff = numericPersonalScore - (stock.personal_opinion_score || 0);
      const updatedStock = {
        ...stock,
        personal_opinion_score: numericPersonalScore,
        total_score: stock.total_score + scoreDiff
      };

      setStock(updatedStock);
      onUpdate(updatedStock);

      const response = await stockPicksApi.updatePersonalScore(stock.id, numericPersonalScore);

      // Update with the actual server response
      setStock(response.data);
      onUpdate(response.data);
      setIsEditingPersonalScore(false);
    } catch (err) {
      console.error('Error updating personal opinion score:', err);
      setError('Failed to update personal opinion score');

      // Revert optimistic update on error
      const response = await stockPicksApi.getStockPick(stock.id);
      setStock(response.data);
      onUpdate(response.data);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Check if a characteristic is selected (considering both actual state and pending changes)
  const isCharacteristicSelected = (characteristicId: number) => {
    // If there's a pending change, use that
    if (characteristicId in pendingCharacteristics) {
      return pendingCharacteristics[characteristicId];
    }
    // Otherwise use the actual state
    return stock.characteristics.some(c => c.characteristic_id === characteristicId);
  };

  // Get characteristic data by ID
  const getCharacteristicById = (characteristicId: number) => {
    return stock.characteristics.find(c => c.characteristic_id === characteristicId);
  };

  // Sort global characteristics based on the ORDERED_CHARACTERISTICS list
  const getSortedGlobalCharacteristics = () => {
    // Sort global characteristics based on dynamic orderedMeta list
    if (!orderedMeta.length) return [...globalCharacteristics];
    const orderMap = new Map(orderedMeta.map(o => [o.name, o.position] as [string, number]));
    return [...globalCharacteristics].sort((a, b) => {
      const posA = orderMap.get(a.name) ?? Number.MAX_SAFE_INTEGER;
      const posB = orderMap.get(b.name) ?? Number.MAX_SAFE_INTEGER;
      if (posA === posB) return a.name.localeCompare(b.name);
      return posA - posB;
    });
  };

  // Generate and download characteristics JSON
  const handleDownloadJson = () => {
    try {
      // Create an object to map each characteristic to its checked status
      const characteristicsStatus: Record<string, boolean> = {};

      // Add entries for all global characteristics
      globalCharacteristics.forEach(gc => {
        const isChecked = isCharacteristicSelected(gc.id);
        characteristicsStatus[gc.name] = isChecked;
      });

      // Create the data object to download
      const downloadData = {
        symbol: stock.symbol,
        total_score: stock.total_score,
        personal_opinion_score: stock.personal_opinion_score,
        details: detailsText,
        demand_reason: stock.demand_reason,
        note: note, // Add this line
        characteristics: characteristicsStatus
      };

      // Convert to JSON
      const jsonData = JSON.stringify(downloadData, null, 2);

      // Create and trigger download
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${stock.symbol}_characteristics.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading JSON:', err);
      setError('Failed to download characteristics data');
    }
  };

  return (
    <Card className="rounded-sm">
      <CardContent className="p-0 pl-1 rounded-sm">
        <div
          className="flex justify-between items-center cursor-pointer p-0.1 overflow-hidden"
          onClick={() => setIsExpanded(!isExpanded)}
          ref={headerRef}
        >
          <div className="flex items-center gap-1 min-w-0">
            <Badge variant="default" className="font-semibold">{stock.symbol}</Badge>
            <Badge variant="secondary" className="whitespace-nowrap">Score: {stock.total_score}</Badge>
            {stock.personal_opinion_score > 0 && (
              <Badge variant="outline" className="bg-green-100/70 dark:bg-green-900/70 whitespace-nowrap">Personal: +{stock.personal_opinion_score}</Badge>
            )}
            {stock.personal_opinion_score < 0 && (
              <Badge variant="outline" className="bg-red-100/70 dark:bg-red-900/70 whitespace-nowrap">Personal: {stock.personal_opinion_score}</Badge>
            )}
            <span ref={catalystWrapperRef} className={hideCatalyst ? 'hidden' : ''}>
              {stock.demand_reason && (
                <Badge variant="outline" className="text-green-600 dark:text-green-400 whitespace-nowrap">{stock.demand_reason}</Badge>
              )}
            </span>
            <span ref={noteWrapperRef} className={hideNote ? 'hidden' : ''}>
              {stock.note && (
                <Badge variant="outline" className="text-blue-600 dark:text-blue-400 whitespace-nowrap">{stock.note}</Badge>
              )}
            </span>
            <span ref={priorityCharsWrapperRef} className={hidePriorityChars ? 'hidden' : 'flex flex-nowrap'}>
              {priorityCharacteristics.length > 0 && priorityCharacteristics.map(char => (
                <Badge key={char.id} variant="outline" className="text-yellow-600 dark:text-yellow-400 whitespace-nowrap">{char.name}</Badge>
              ))}
            </span>

            {/* Regular Characteristics container with ref for measuring */}
            <span ref={regularCharsWrapperRef} className={hideRegularChars ? 'hidden' : ''}>
              <div className="flex flex-wrap gap-1 max-h-7 overflow-hidden" ref={charContainerRef}>
                {visibleCharacteristics.map((char) => (
                  <Badge key={char.id} variant="outline">
                    {char.name}
                  </Badge>
                ))}

                {hasHiddenCharacteristics && (
                  <Badge variant="outline" className="flex items-center">
                    <MoreHorizontal className="h-3 w-3 mr-0.5" />
                  </Badge>
                )}
              </div>
            </span>

            {/* Personal badge moved earlier */}
          </div>

          <div className="flex items-center gap-1 flex-shrink-0">
            {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="p-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (hasTradeInHistory === null) {
                      // Fire async check; no need to await
                      checkTradeExists();
                    }
                  }}
                >
                  <X className="h-3 w-3 text-destructive" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Remove {stock.symbol}</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to remove this stock from your ranking? This action cannot be undone.
                    {!hasTradeInHistory && hasTradeInHistory !== null && (
                      <div className="mt-3 flex items-start gap-2 text-amber-600 dark:text-amber-400 text-sm">
                        <AlertTriangle className="h-4 w-4 mt-0.5" />
                        <span>
                          No trade exists in history for <strong>{stock.symbol}</strong> yet. If you add a trade first, the full ranking JSON (characteristics, scores, notes, catalyst, details) will be auto-saved into the trade Case field so you can review it later even after removing this ranking.
                        </span>
                      </div>
                    )}
                    {isCheckingTrades && (
                      <div className="mt-2 text-xs text-muted-foreground">Checking trade history...</div>
                    )}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove();
                    }}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Remove
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>

        {isExpanded && (
          <div className="mt-1 p-2">
            {error && (
              <Alert variant="destructive" className="py-1 px-2 mb-2">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Note field - Add this after the Demand Reason section (around line 624) */}
            <div className="mb-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Note:</span>
                <div className="flex-1">
                  <Input
                    value={note || ''}
                    onFocus={() => setIsEditingNote(true)}
                    onBlur={() => setIsEditingNote(false)}
                    onChange={(e) => {
                      const newNote = e.target.value;
                      setNote(newNote);
                      setStock(prev => ({ ...prev, note: newNote }));
                      persistLocal({ note: newNote });
                      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
                      const currentRequestVersion = ++requestVersionRef.current;
                      saveTimeoutRef.current = setTimeout(async () => {
                        try {
                          // removed saving indicator
                          await stockPicksApi.updateStockPick(stock.id, {
                            note: newNote,
                            demand_reason: stock.demand_reason,
                            ranking_box: stock.ranking_box,
                            symbol: stock.symbol,
                            total_score: stock.total_score,
                            personal_opinion_score: stock.personal_opinion_score,
                            case_text: stock.case_text
                          }).then(resp => {
                            if (currentRequestVersion !== requestVersionRef.current) return;
                            setStock(prev => ({ ...prev, ...resp.data, note: prev.note }));
                            onUpdate({ ...resp.data, note: newNote });
                          });
                        } catch (err) {
                          console.error('Error saving note:', err);
                          setError('Failed to save note');
                        } finally {
                          // removed saving indicator
                        }
                      }, 900);
                    }}
                    placeholder="Enter note..."
                    className="h-8"
                    onKeyDown={(e) => handleManualSaveShortcut(e, 'note')}
                  />
                  {isEditingNote && manualSavingField !== 'note' && (
                    <div className="mt-1 text-xs text-muted-foreground">Press Ctrl+S to save immediately</div>
                  )}
                  {manualSavingField === 'note' && (
                    <div className="mt-1 text-xs text-muted-foreground">Saving...</div>
                  )}
                </div>
              </div>
            </div>

            <div className="mb-4">
              <div className="flex justify-between items-center mb-1">
                <h3 className="text-sm font-medium">Characteristics</h3>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 py-0 flex items-center gap-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownloadJson();
                  }}
                >
                  <Download className="h-3 w-3" />
                  <span className="text-xs">Download</span>
                </Button>
              </div>

              {/* Compact Grid of Global Characteristics - sorted by predefined order */}
              {metaError && (
                <div className="text-xs text-red-500 mb-1">{metaError}</div>
              )}
              {metaLoading && (
                <div className="text-xs text-muted-foreground mb-1">Loading lists...</div>
              )}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-0.5">
                {getSortedGlobalCharacteristics().map(globalChar => {
                  const isSelected = isCharacteristicSelected(globalChar.id);
                  const selectedChar = isSelected ? getCharacteristicById(globalChar.id) : null;
                  const isColorCoded = colorCodedMeta.some(cc => cc.name === globalChar.name);

                  return (
                    <div
                      key={globalChar.id}
                      className={`flex items-center justify-between border p-0.5 rounded min-h-[28px] 
                        ${lastClickedCharacteristic === globalChar.id ? 'bg-primary/20' : ''} 
                        ${isColorCoded ?
                          (isSelected
                            ? 'bg-green-100 dark:bg-green-800/40'
                            : 'bg-red-50 dark:bg-red-900/40 border-red-200 dark:border-red-950/50'
                          ) : 'hover:bg-secondary/20'
                        }
                        transition-colors duration-150 cursor-default`}
                      onClick={() => setLastClickedCharacteristic(globalChar.id)}
                    >

                      <div className="flex items-center gap-1">
                        <Checkbox
                          checked={isSelected}
                          onCheckedChange={(checked) => handleToggleCharacteristic(globalChar.id, checked === true)}
                          disabled={isSubmitting && !(globalChar.id in pendingCharacteristics)}
                        />
                        <span className="text-sm">{globalChar.name}</span>
                      </div>

                      {isSelected && selectedChar && (
                        <div className="flex items-center gap-0.5">
                          {isEditingScore === globalChar.id ? (
                            <div className="flex items-center gap-0.5">
                              <Input
                                type="text"
                                inputMode="numeric"
                                value={editedScore}
                                onChange={(e) => setEditedScore(e.target.value)}
                                className="h-8 w-24 text-sm"
                                onClick={(e) => e.stopPropagation()}
                              />
                              <Button
                                size="icon"
                                variant="ghost"
                                className="h-5 w-5"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSaveScore(globalChar.id);
                                }}
                                disabled={isSubmitting}
                              >
                                <Check className="h-3 w-3" />
                              </Button>
                            </div>
                          ) : (
                            <>
                              <Badge variant="secondary" className="text-xs px-1">
                                {selectedChar.score}
                              </Badge>
                              <Button
                                size="icon"
                                variant="ghost"
                                className="h-5 w-5 p-0"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleStartEditScore(globalChar.id);
                                }}
                                disabled={isSubmitting}
                              >
                                <Edit className="h-3 w-3" />
                              </Button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            {/* Demand Reason - More compact layout */}
            <div className="mb-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Catalyst:</span>
                <div className="flex-1">
                  <Input
                    value={stock.demand_reason || ''}
                    onFocus={() => setIsEditingDemandReason(true)}
                    onBlur={() => setIsEditingDemandReason(false)}
                    onChange={(e) => {
                      const newDemandReason = e.target.value;
                      setStock(prev => ({ ...prev, demand_reason: newDemandReason }));
                      persistLocal({ demand_reason: newDemandReason });
                      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
                      const currentRequestVersion = ++requestVersionRef.current;
                      saveTimeoutRef.current = setTimeout(async () => {
                        try {
                          // removed saving indicator
                          await stockPicksApi.updateStockPick(stock.id, {
                            demand_reason: newDemandReason,
                            ranking_box: stock.ranking_box,
                            symbol: stock.symbol,
                            total_score: stock.total_score,
                            personal_opinion_score: stock.personal_opinion_score,
                            case_text: stock.case_text,
                            note: stock.note
                          }).then(resp => {
                            if (currentRequestVersion !== requestVersionRef.current) return;
                            setStock(prev => ({ ...prev, ...resp.data, demand_reason: prev.demand_reason }));
                            onUpdate({ ...resp.data, demand_reason: newDemandReason });
                          });
                        } catch (err) {
                          console.error('Error saving demand reason:', err);
                          setError('Failed to save demand reason');
                        } finally {
                          // removed saving indicator
                        }
                      }, 900);
                    }}
                    placeholder="Enter demand reason..."
                    className="h-8"
                    onKeyDown={(e) => handleManualSaveShortcut(e, 'catalyst')}
                  />
                  {isEditingDemandReason && manualSavingField !== 'catalyst' && (
                    <div className="mt-1 text-xs text-muted-foreground">Press Ctrl+S to save immediately</div>
                  )}
                  {manualSavingField === 'catalyst' && (
                    <div className="mt-1 text-xs text-muted-foreground">Saving...</div>
                  )}
                </div>
              </div>
            </div>


            {/* Personal Opinion Score - More compact layout */}
            <div className="mb-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Personal Opinion Score:</span>
                {isEditingPersonalScore ? (
                  <>
                    <Input
                      type="text"
                      inputMode="numeric"
                      value={personalScore}
                      onChange={(e) => setPersonalScore(e.target.value)}
                      className="w-24 h-8"
                      onClick={(e) => e.stopPropagation()}
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSavePersonalScore();
                      }}
                      disabled={isSubmitting}
                      className="h-8"
                    >
                      <Check className="h-4 w-4 mr-1" />
                      Save
                    </Button>
                  </>
                ) : (
                  <>
                    <Badge
                      variant={Number(personalScore) === 0 ? "secondary" : Number(personalScore) > 0 ? "outline" : "destructive"}
                      className={Number(personalScore) > 0 ? "bg-green-100/70 dark:bg-green-900/70 px-2" : "px-2"}
                    >
                      {Number(personalScore) > 0 ? `+${personalScore}` : personalScore}
                    </Badge>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleStartEditPersonalScore();
                      }}
                      disabled={isSubmitting}
                      className="h-8"
                    >
                      <Edit className="h-4 w-4 mr-1" />
                      Edit
                    </Button>
                  </>
                )}
              </div>
            </div>

            {/* More Details */}
            <div>
              <span className="text-sm font-medium">Details</span>
              <Textarea
                placeholder="Enter additional details..."
                value={detailsText}
                onFocus={() => setIsEditingCaseText(true)}
                onBlur={() => setIsEditingCaseText(false)}
                onChange={handleDetailsTextChange}
                onKeyDown={(e) => handleManualSaveShortcut(e, 'details')}
                className="mt-1"
                rows={4}
                style={{ resize: 'vertical' }}
              />
              {isEditingCaseText && manualSavingField !== 'details' && (
                <div className="mt-1 text-xs text-muted-foreground">Press Ctrl+S to save immediately</div>
              )}
              {manualSavingField === 'details' && (
                <div className="mt-1 text-xs text-muted-foreground">Saving...</div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RankingItem;
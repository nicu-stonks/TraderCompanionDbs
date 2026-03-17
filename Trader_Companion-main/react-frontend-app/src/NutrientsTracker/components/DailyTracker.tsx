import { useState, useEffect, useRef, memo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { fetchDailyRecord, sendChat, getChatPrompt, submitManualChat, DailyRecord, fetchConfig, LLMConfig, deleteDailyRecord, saveConfig, rollbackChatBeforeMessage } from '../services/api';
import { NutrientStats } from './NutrientStats';
import { Trash2, Copy } from 'lucide-react';

const CHAT_LOCK_TTL_MS = 5 * 60 * 1000;

const ChatHistoryList = memo(({
  history,
  onDiscardFromMessage,
  isDiscarding,
}: {
  history: any[];
  onDiscardFromMessage: (index: number) => void;
  isDiscarding: boolean;
}) => {
  return (
    <>
      {history.map((msg, i) => (
        <div key={i} className={`flex ${msg.role === 'model' ? 'justify-start' : 'justify-end'}`}>
          <div className={`max-w-[100%] sm:max-w-[90%] p-4 rounded-xl ${msg.role === 'model' ? 'bg-secondary text-secondary-foreground shadow-sm' : 'bg-primary text-primary-foreground shadow-sm'}`}>
            <div className="text-xs font-bold mb-2 opacity-75">{msg.role === 'model' ? 'AI Nutritionist' : 'You'}</div>
            <div className="text-sm">
              {msg.role === 'model' ? (
                <div className="prose dark:prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-headings:font-bold prose-headings:mt-6 prose-headings:mb-3 prose-ul:my-2 prose-li:my-0">
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                    {msg.parts.map((p: any) => p.text).join('').replace(/\n*(#{1,6}\s)/g, (_match: string, p1: string, offset: number) => (offset === 0 ? p1 : '\n\n&nbsp;\n\n' + p1))}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.parts.map((p: any) => p.text).join('')}</div>
              )}
            </div>
            {msg.role === 'user' && (
              <div className="mt-3 flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={isDiscarding}
                  onClick={() => onDiscardFromMessage(i)}
                >
                  Discard from here
                </Button>
              </div>
            )}
          </div>
        </div>
      ))}
    </>
  );
});

export function DailyTracker() {
  const [date, setDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [record, setRecord] = useState<DailyRecord | null>(null);
  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [selectedModel, setSelectedModel] = useState('');

  const [prompt, setPrompt] = useState('');
  const [apiMode, setApiMode] = useState<'api' | 'manual'>('api');
  const [manualPrompt, setManualPrompt] = useState('');
  const [manualResponse, setManualResponse] = useState('');
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false);
  const [isDiscarding, setIsDiscarding] = useState(false);

  const [isTyping, setIsTyping] = useState(() => {
    const lockTime = localStorage.getItem('chat_lock');
    if (!lockTime) {
      return false;
    }

    const lockAge = Date.now() - parseInt(lockTime, 10);
    if (Number.isNaN(lockAge) || lockAge > CHAT_LOCK_TTL_MS) {
      localStorage.removeItem('chat_lock');
      return false;
    }

    return true;
  });
  const [error, setError] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const prevChatCountRef = useRef<number>(0);
  const initialLoadDoneRef = useRef<boolean>(false);
  const isUnloadingRef = useRef<boolean>(false);

  useEffect(() => {
    const handleBeforeUnload = () => { isUnloadingRef.current = true; };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  // Periodic TTL check: auto-clear isTyping if the lock has expired
  useEffect(() => {
    if (!isTyping) return;
    const intervalId = setInterval(() => {
      const lockTime = localStorage.getItem('chat_lock');
      if (!lockTime) {
        setIsTyping(false);
        return;
      }
      const lockAge = Date.now() - parseInt(lockTime, 10);
      if (Number.isNaN(lockAge) || lockAge > CHAT_LOCK_TTL_MS) {
        localStorage.removeItem('chat_lock');
        setIsTyping(false);
      }
    }, 10_000); // check every 10 seconds
    return () => clearInterval(intervalId);
  }, [isTyping]);

  useEffect(() => {
    fetchConfig().then(c => {
      setConfig(c);
      if (c && c.model_name) {
        setSelectedModel(c.model_name);
      }
    }).catch(console.error);
  }, []);

  useEffect(() => {
    let active = true;
    const loadRecord = () => {
      fetchDailyRecord(date)
        .then(res => {
          if (active) {
            setRecord(res);
          }
        })
        .catch(() => {
          if (active) {
            setRecord(null);
          }
        });
    };

    // Initial load
    loadRecord();

    // Poll every 1 second
    const intervalId = setInterval(loadRecord, 1000);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, [date]);

  useEffect(() => {
    // Scroll chat to bottom only when a new message arrives or user is typing
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [record?.chat_history?.length, isTyping, error]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [prompt]);

  useEffect(() => {
    const chatHistory = record?.chat_history;
    if (chatHistory) {
      if (initialLoadDoneRef.current) {
        if (chatHistory.length > prevChatCountRef.current && isTyping) {
          // A new message arrived via polling while 'typing'
          setIsTyping(false);
          localStorage.removeItem('chat_lock');
        }
      } else {
        initialLoadDoneRef.current = true;
      }
      prevChatCountRef.current = chatHistory.length;
    }
  }, [record?.chat_history, isTyping]);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    setIsTyping(true);
    localStorage.setItem('chat_lock', Date.now().toString());
    setError(null);
    let requestFailed = false;
    let failedToFetch = false;

    try {
      const updated = await sendChat(date, prompt, selectedModel);
      setRecord(updated);
      setPrompt('');
    } catch (e: any) {
      requestFailed = true;
      failedToFetch = e.message === 'Failed to fetch';
      setError(e.message || "Failed to contact LLM");
    } finally {
      if (!isUnloadingRef.current && requestFailed && !failedToFetch) {
        setIsTyping(false);
        localStorage.removeItem('chat_lock');
      }
    }
  };

  const handleGeneratePrompt = async () => {
    if (!prompt.trim()) return;
    setIsGeneratingPrompt(true);
    setError(null);
    try {
      const res = await getChatPrompt(date, prompt);
      setManualPrompt(res.full_prompt);
    } catch (e: any) {
      setError(e.message || "Failed to generate prompt");
    } finally {
      setIsGeneratingPrompt(false);
    }
  };

  const handleManualSubmit = async () => {
    if (!manualResponse.trim()) return;
    setIsTyping(true);
    setError(null);
    try {
      const updated = await submitManualChat(date, prompt, manualResponse);
      setRecord(updated);
      setPrompt('');
      setManualPrompt('');
      setManualResponse('');
    } catch (e: any) {
      setError(e.message || "Failed to submit manual response");
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearDay = async () => {
    if (window.confirm(`Are you sure you want to completely erase all chat, foods, and nutrients tracked for ${date}? This cannot be undone.`)) {
      try {
        await deleteDailyRecord(date);
        setRecord(null);
        setPrompt('');
      } catch (e) {
        console.error(e);
        alert("Failed to clear day.");
      }
    }
  };

  const handleDiscardFromMessage = async (messageIndex: number) => {
    if (!window.confirm('Discard this message and everything after it? This cannot be undone.')) {
      return;
    }

    setIsDiscarding(true);
    setError(null);
    try {
      const res = await rollbackChatBeforeMessage(date, messageIndex);
      setRecord(res.record);
      setIsTyping(false);
      localStorage.removeItem('chat_lock');
    } catch (e: any) {
      setError(e.message || 'Failed to rollback chat');
    } finally {
      setIsDiscarding(false);
    }
  };

  const handleModelChange = async (newModel: string) => {
    setSelectedModel(newModel);
    if (config) {
      const updatedConfig = { ...config, model_name: newModel };
      setConfig(updatedConfig);
      try {
        await saveConfig(updatedConfig);
      } catch (e) {
        console.error("Failed to save model preference", e);
      }
    }
  };

  const nutrientBars = record?.nutrient_completion || {};

  return (
    <div className="space-y-6">
      {/* Top Row: Full Width Progress Bars */}
      <Card>
        <CardHeader>
          <CardTitle>Daily Completion</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.keys(nutrientBars).length === 0 ? (
            <div className="text-sm text-muted-foreground italic">No data yet. Say what you ate today to start tracking!</div>
          ) : (
            <NutrientStats completionData={nutrientBars} sourcesData={record?.nutrient_sources || {}} />
          )}
        </CardContent>
      </Card>

      {/* Bottom Row: 4-Column Grid for Chat and Logs */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-stretch min-h-[500px] h-full">

        {/* Left Column: Foods & Supplements */}
        <div className="lg:col-span-1 flex flex-col gap-6 min-h-0">

          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader>
              <CardTitle>Foods Tracked</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 min-h-0 pb-4">
              <div className="text-base border rounded p-3 h-full overflow-y-auto whitespace-pre-wrap bg-muted/20">
                {record?.foods_eaten ? (
                  <ul className="list-disc list-inside space-y-0.5 leading-tight">
                    {record.foods_eaten.split('\n').map((food: string, i: number) => (
                      <li key={i}>{food.replace(/^- /, '')}</li>
                    ))}
                  </ul>
                ) : (
                  <span className="text-muted-foreground italic">No foods logged yet.</span>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader>
              <CardTitle>Supplements Tracked</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 min-h-0 pb-4">
              <div className="text-base border rounded p-3 h-full overflow-y-auto whitespace-pre-wrap bg-muted/20">
                {record?.supplements_taken && record.supplements_taken.length > 0 ? (
                  <ul className="list-disc list-inside space-y-0.5 leading-tight">
                    {record.supplements_taken.map((supp: string, i: number) => (
                      <li key={i}>{supp}</li>
                    ))}
                  </ul>
                ) : (
                  <span className="text-muted-foreground italic">No supplements logged yet.</span>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Chat interface */}
        <div className="lg:col-span-3 flex flex-col min-h-0 h-full">
          <Card className="flex-1 flex flex-col min-h-0 h-full">
            <CardHeader>
              <CardTitle className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                <span>AI Nutritionist Chat</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground font-normal">Date:</span>
                  <input
                    type="date"
                    value={date}
                    onChange={e => setDate(e.target.value)}
                    className="text-sm bg-transparent border rounded p-1"
                  />
                  <Button variant="destructive" size="icon" className="h-8 w-8 ml-2" onClick={handleClearDay} title="Clear Day">
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col space-y-4 min-h-0 h-full">

              {/* Chat History */}
              <div className="h-[60vh] overflow-y-auto border rounded-md p-4 space-y-4 bg-muted/20 relative">
                {record?.chat_history && (
                  <ChatHistoryList
                    history={record.chat_history}
                    onDiscardFromMessage={handleDiscardFromMessage}
                    isDiscarding={isDiscarding}
                  />
                )}
                {(!record || !record.chat_history || record.chat_history.length === 0) && !isTyping && (
                  <div className="text-sm text-muted-foreground italic flex h-full justify-center items-center">
                    Welcome! Start by telling me what you ate today, or ask me for a recommendation.
                  </div>
                )}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] p-3 rounded-lg bg-secondary text-secondary-foreground text-sm italic">
                      AI is typing and analyzing your intake...
                    </div>
                  </div>
                )}
                {error && (
                  <div className="flex justify-center p-2 text-sm text-red-500 font-semibold bg-red-100 dark:bg-red-900 rounded">
                    Error: {error}
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Manual Mode Additional Inputs */}
              {apiMode === 'manual' && (
                <div className="flex flex-col gap-3 p-4 bg-muted/10 rounded border border-dashed text-sm">
                  <div className="font-semibold text-muted-foreground">Manual Mode Workflow:</div>
                  <div className="text-xs text-muted-foreground">
                    1. Type your prompt below and click "Generate Prompt".<br />
                    2. Copy the generated text and paste it into ChatGPT/Claude.<br />
                    3. Copy their exact markdown response and paste it into the "Paste Response" box below.<br />
                    4. Click "Submit Manual Response".
                  </div>

                  {manualPrompt && (
                    <div className="space-y-2 mt-2">
                      <div className="flex justify-between items-center">
                        <span className="font-bold">Generated Prompt</span>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => navigator.clipboard.writeText(manualPrompt)}>
                          <Copy className="w-3 h-3 mr-1" /> Copy
                        </Button>
                      </div>
                      <Textarea value={manualPrompt} readOnly className="h-32 text-xs font-mono bg-muted" />
                    </div>
                  )}

                  {manualPrompt && (
                    <div className="space-y-2 mt-2">
                      <span className="font-bold">Paste LLM Response</span>
                      <Textarea
                        value={manualResponse}
                        onChange={e => setManualResponse(e.target.value)}
                        placeholder="Paste the ChatGPT/Claude response here including the ```json block..."
                        className="h-32 text-xs font-mono"
                      />
                      <Button
                        className="w-full mt-2"
                        onClick={handleManualSubmit}
                        disabled={!manualResponse.trim() || isTyping}
                      >
                        Submit Manual Response
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {/* Input Area */}
              <div className="flex gap-2 relative">
                <div className="flex flex-col w-[135px] shrink-0 gap-2">
                  <select
                    className="text-sm border rounded px-2 py-1 bg-background w-full cursor-pointer focus:ring-1"
                    value={apiMode}
                    onChange={e => {
                      setApiMode(e.target.value as 'api' | 'manual');
                      setManualPrompt('');
                      setManualResponse('');
                    }}
                  >
                    <option value="api">API Mode</option>
                    <option value="manual">Manual Mode</option>
                  </select>

                  <select
                    className="text-sm border rounded px-2 py-1 bg-background w-full cursor-pointer focus:ring-1 disabled:opacity-50 truncate"
                    value={selectedModel}
                    onChange={e => handleModelChange(e.target.value)}
                    disabled={apiMode === 'manual'}
                    title={selectedModel}
                  >
                    {!config?.available_models && <option value={selectedModel}>{selectedModel}</option>}
                    {config?.available_models && config.available_models.split(',').map(m => m.trim()).filter(m => m).map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
                <Textarea
                  ref={textareaRef}
                  placeholder="E.g. I had 3 eggs for breakfast, a salmon salad, and took 1 omega 3 pill..."
                  className="flex-grow resize-none overflow-hidden max-h-[300px]"
                  rows={2}
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (apiMode === 'api') handleSend();
                      else handleGeneratePrompt();
                    }
                  }}
                />
                <Button
                  className="h-auto self-stretch"
                  onClick={apiMode === 'api' ? handleSend : handleGeneratePrompt}
                  disabled={isTyping || isDiscarding || !prompt.trim() || (apiMode === 'manual' && isGeneratingPrompt)}
                >
                  {apiMode === 'api' ? 'Send' : 'Generate Prompt'}
                </Button>
              </div>
              <div className="text-xs text-muted-foreground italic text-center">
                Press Enter to send. Shift+Enter for newline.
                <br />
                <b>Avoid spamming the Send button to prevent API rate limits.</b>
              </div>

            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { askQuestion, fetchSchemes } from "./api";
import AnswerView from "./components/AnswerView";
import type { Answer, Scheme } from "./types";

const EXAMPLES = [
  "Who is eligible for DPIIT startup recognition?",
  "What is the maximum loan under MUDRA Kishore?",
  "How much seed funding does SISFS provide?",
  "What is the Udyam registration threshold for a small enterprise?",
  "Who can apply for a Stand-Up India loan?",
  "What collateral-free cover does CGTMSE offer?",
];

export default function App() {
  const [schemes, setSchemes] = useState<Scheme[]>([]);
  const [scheme, setScheme] = useState("all");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSchemes()
      .then(setSchemes)
      .catch(() => setError("Could not load the list of schemes."));
  }, []);

  async function ask(q: string) {
    const text = q.trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await askQuestion(text, scheme));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    ask(question);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(question);
    }
  }

  function pickExample(q: string) {
    setQuestion(q);
    ask(q);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-5">
          <h1 className="text-2xl font-bold text-indigo-800">SchemeSetu</h1>
          <p className="mt-1 text-sm text-slate-600">
            Ask about Indian startup and MSME schemes and get answers from the official documents.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl space-y-6 px-4 py-6">
        <form onSubmit={onSubmit} className="space-y-3">
          <div>
            <label htmlFor="scheme" className="block text-sm font-medium text-slate-700">
              Scheme
            </label>
            <select
              id="scheme"
              value={scheme}
              onChange={(e) => setScheme(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm sm:w-72"
            >
              <option value="all">All schemes</option>
              {schemes.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="question" className="block text-sm font-medium text-slate-700">
              Your question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onKeyDown}
              maxLength={500}
              rows={3}
              placeholder="e.g. Who is eligible for a MUDRA loan?"
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="w-full rounded-md bg-indigo-700 px-5 py-2 text-sm font-semibold text-white hover:bg-indigo-800 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
          >
            {loading ? "Searching..." : "Ask"}
          </button>
        </form>

        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Try an example</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => pickExample(ex)}
                disabled={loading}
                className="rounded-full border border-slate-300 bg-white px-3 py-1 text-left text-xs text-slate-700 hover:border-indigo-400 hover:text-indigo-800 disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex items-center gap-3 text-sm text-slate-600" role="status">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-200 border-t-indigo-700" />
            Searching the scheme documents...
          </div>
        )}

        {answer && <AnswerView answer={answer} />}
      </main>
    </div>
  );
}

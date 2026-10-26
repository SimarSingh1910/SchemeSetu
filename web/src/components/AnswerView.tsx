import type { Answer } from "../types";

function SourceCard({ section }: { section: Answer["sources"][number] }) {
  const pages = section.pages.length
    ? ` - page${section.pages.length > 1 ? "s" : ""} ${section.pages.join(", ")}`
    : "";
  const title = `From ${section.source_doc}${section.scheme ? ` - ${section.scheme}` : ""}${pages}`;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="break-words text-sm font-semibold text-indigo-800">{title}</h3>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-700">
        {section.points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
    </section>
  );
}

export default function AnswerView({ answer }: { answer: Answer }) {
  if (!answer.found) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-900">
        <p className="font-semibold">Not found in the scheme documents</p>
        <p className="mt-1 text-sm">{answer.summary}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-indigo-50 p-4 text-slate-800">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-indigo-700">Summary</h2>
        <p className="mt-1">{answer.summary}</p>
      </div>
      {answer.mode === "extractive" && (
        <p className="text-xs text-slate-500">
          These are excerpts taken directly from the documents; a generated summary was not available
          for this question.
        </p>
      )}
      {answer.sources.map((s) => (
        <SourceCard key={s.source_doc} section={s} />
      ))}
    </div>
  );
}

export interface Scheme {
  id: string;
  name: string;
  docs: string[];
}

export interface SourceSection {
  source_doc: string;
  scheme: string;
  pages: number[];
  points: string[];
}

export interface Answer {
  question: string;
  scheme: string | null;
  summary: string;
  sources: SourceSection[];
  found: boolean;
  mode: "generated" | "extractive";
}

import { lazy, type ComponentType, type LazyExoticComponent } from "react";

/** Every project in the portfolio. Add one: an entry here + its page in projects/<slug>/ (+ its API in app.py). */
export type ProjectInfo = {
  slug: string;                 // URL /projects/<slug>; also the API prefix /api/<slug>
  title: string;
  tagline: string;
  tags: string[];
  stat: { label: string; value: string };
  Page: LazyExoticComponent<ComponentType>;
};

export const REPO_URL = "https://github.com/mazprimrez/project-playground";

export const PROJECTS: ProjectInfo[] = [
  {
    slug: "genrec",
    title: "GenRec: an LLM that recommends movies",
    tagline:
      "Qwen2.5-0.5B fine-tuned to know 4,000 recent movies (2010-2023) and predict what a user watches next, " +
      "compared with SASRec, TensorFlow Recommenders and simple baselines.",
    tags: ["LLM fine-tuning", "Recommender systems", "PyTorch", "FastAPI"],
    stat: { label: "HR@10 on 1,000 test users", value: "LLM 0.139 · SASRec 0.182 · popularity 0.088" },
    Page: lazy(() => import("./genrec/GenrecPage")),
  },
];

import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";
import { factorBaseModel, resolveModelMetadataBaseModel } from "./openrouter.js";

// Public, unauthenticated. Every served model carries a `metadata` block with
// pricing, capabilities, limits, and deprecation state.
const API_ENDPOINT = "https://api.neuralwatt.com/v1/models";

const NeuralwattPricing = z.object({
  input_per_million: z.number().nonnegative().nullish(),
  output_per_million: z.number().nonnegative().nullish(),
  cached_input_per_million: z.number().nonnegative().nullish(),
  pricing_tbd: z.boolean().nullish(),
}).passthrough();

const NeuralwattMetadata = z.object({
  display_name: z.string().nullish(),
  description: z.string().nullish(),
  huggingface_id: z.string().nullish(),
  pricing: NeuralwattPricing.nullish(),
  capabilities: z.object({
    tools: z.boolean().nullish(),
    vision: z.boolean().nullish(),
    reasoning: z.boolean().nullish(),
    streaming: z.boolean().nullish(),
  }).passthrough().nullish(),
  limits: z.object({
    max_context_length: z.number().int().positive().nullish(),
    max_output_tokens: z.number().int().positive().nullish(),
  }).passthrough().nullish(),
  deprecated: z.boolean().nullish(),
}).passthrough();

export const NeuralwattModel = z.object({
  id: z.string().min(1),
  object: z.literal("model"),
  max_model_len: z.number().int().positive().nullish(),
  metadata: NeuralwattMetadata.nullish(),
}).passthrough();

export const NeuralwattResponse = z.object({
  object: z.literal("list"),
  data: z.array(NeuralwattModel),
}).passthrough();

export type NeuralwattModel = z.infer<typeof NeuralwattModel>;

// Flex-tier aliases are billed at a runtime-configured discount that only
// applies to requests that actually take the flex path (streaming required;
// non-streaming falls through to the standard tier and standard pricing). The
// API therefore advertises the standard price on flex aliases. The discounted
// price in these TOMLs is hand-authored and preserved by the sync.
function isFlex(id: string) {
  return id.endsWith("-flex");
}

export const neuralwatt = {
  id: "neuralwatt",
  name: "Neuralwatt",
  modelsDir: "providers/neuralwatt/models",
  // Models leave the live catalog deliberately (deprecation flow); retain the
  // local TOML and surface it for manual lifecycle review rather than
  // auto-deleting on an API blip.
  deleteMissing: false,
  sourceID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `${ids.length} Neuralwatt models were not created automatically: flex-tier aliases need a hand-authored discounted cost (the API advertises the standard, fall-through price), and other skips lacked a resolvable base model or complete pricing.`,
      `Skipped remote IDs: ${ids.map((id) => `\`${id}\``).join(", ")}`,
    ];
  },
  missingNotice(paths) {
    if (paths.length === 0) return [];
    return [
      `${paths.length} local Neuralwatt models were absent from the live \`/v1/models\` catalog and were retained for manual lifecycle review.`,
      `Retained local paths: ${paths.map((item) => `\`${item}\``).join(", ")}`,
    ];
  },
  async fetchModels() {
    return fetchNeuralwattModels();
  },
  parseModels(raw) {
    return NeuralwattResponse.parse(raw).data;
  },
  translateModel(model, context) {
    const id = model.id;
    const existing = context.existing(id);

    if (existing === undefined) {
      // Backend model IDs (e.g. `deepseek-ai/DeepSeek-V4-Flash`) duplicate an
      // alias entry; don't author new files for them.
      if (id.includes("/")) return undefined;
      if (model.metadata?.deprecated === true) return undefined;
      // Flex aliases need a hand-authored discounted cost (see isFlex note).
      if (isFlex(id)) return undefined;
    }

    const translated = buildNeuralwattModel(model, existing);
    if (translated === undefined) return undefined;
    return { id, model: translated };
  },
} satisfies SyncProvider<NeuralwattModel>;

export async function fetchNeuralwattModels(fetcher: typeof fetch = fetch) {
  const response = await fetcher(API_ENDPOINT);
  if (!response.ok) {
    throw new Error(`Neuralwatt models request failed: ${response.status} ${response.statusText}`);
  }
  return NeuralwattResponse.parse(await response.json());
}

export function buildNeuralwattModel(
  model: NeuralwattModel,
  existing: ExistingModel | undefined,
): SyncedModel | undefined {
  const metadata = model.metadata;
  const pricing = metadata?.pricing;

  const contextLength = metadata?.limits?.max_context_length ?? model.max_model_len ?? undefined;
  const limit = contextLength === undefined
    ? existing?.limit
    : {
      ...existing?.limit,
      context: contextLength,
      output: metadata?.limits?.max_output_tokens ?? existing?.limit?.output ?? contextLength,
    };

  const syncedCost = pricing?.pricing_tbd !== true
    && pricing?.input_per_million !== undefined && pricing?.input_per_million !== null
    && pricing?.output_per_million !== undefined && pricing?.output_per_million !== null
    ? {
      input: pricing.input_per_million,
      output: pricing.output_per_million,
      cache_read: pricing.cached_input_per_million ?? undefined,
    }
    : undefined;

  // Flex aliases keep their hand-authored discounted cost; the API's number is
  // the standard (fall-through) price.
  const cost = isFlex(model.id)
    ? existing?.cost
    : syncedCost ?? existing?.cost;

  if (existing !== undefined) {
    const { base_model: baseModel, base_model_omit: baseModelOmit, ...current } = existing;
    const values = {
      ...current,
      ...(cost === undefined ? {} : { cost }),
      ...(limit === undefined ? {} : { limit }),
    } as SyncedFullModel;
    return baseModel === undefined
      ? values
      : factorBaseModel(baseModel, values, limit, baseModelOmit);
  }

  // Brand-new alias: author it only when a canonical base model exists to
  // inherit curated metadata from, and the API supplied a usable price.
  const hfID = metadata?.huggingface_id ?? undefined;
  const baseModel = hfID === undefined ? undefined : resolveModelMetadataBaseModel(hfID);
  if (baseModel === undefined || cost === undefined || limit === undefined) return undefined;

  const values: Parameters<typeof factorBaseModel>[1] = {
    cost,
    limit,
  };
  const name = metadata?.display_name ?? undefined;
  if (name !== undefined) values.name = name;
  return factorBaseModel(baseModel, values, limit);
}

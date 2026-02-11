from dataclasses import dataclass
from typing import List, Optional, Tuple

from lm_eval.api.model import LM

from eval_runner.model import GenerateParams, LocalGatewayModel


@dataclass(frozen=True)
class Gen:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 64
    stop: Optional[List[str]] = None


class GatewayHarnessLM(LM):
    def __init__(self, model: str = "llama3:latest") -> None:
        super().__init__()
        self.model = model
        self.gw = LocalGatewayModel()

    @property
    def eot_token_id(self) -> int:
        return 0

    @property
    def max_length(self) -> int:
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return 256

    def tok_encode(self, string: str) -> List[int]:
        return [0] * max(1, len(string))

    def tok_decode(self, tokens: List[int]) -> str:
        return ""

    def generate_until(self, requests) -> List[str]:
        outs: List[str] = []
        for inst in requests:
            prompt, gen_kwargs = inst.args
            g = Gen(
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                top_p=float(gen_kwargs.get("top_p", 1.0)),
                top_k=int(gen_kwargs.get("top_k", 40)),
                max_tokens=int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_tokens", 64))),
                stop=gen_kwargs.get("until", None),
            )
            params = GenerateParams(
                model=self.model,
                temperature=g.temperature,
                top_p=g.top_p,
                top_k=g.top_k,
                max_tokens=g.max_tokens,
                response_format="text",
            )
            rec = self.gw.generate(prompt, params=params)
            outs.append((rec.get("response", {}).get("output") or "").strip())
        return outs

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        out: List[Tuple[float, bool]] = []
        for inst in requests:
            context, continuation = inst.args
            continuation = continuation or ""
            if continuation == "":
                out.append((0.0, True))
                continue

            params = GenerateParams(
                model=self.model,
                temperature=0.0,
                top_p=1.0,
                top_k=40,
                max_tokens=max(8, min(64, len(continuation) + 8)),
                response_format="text",
            )
            rec = self.gw.generate(context, params=params)
            gen = (rec.get("response", {}).get("output") or "")
            gen_s = gen.lstrip()
            cont_s = continuation.lstrip()

            if gen_s.startswith(cont_s):
                out.append((0.0, True))
            else:
                out.append((-1.0 * float(len(cont_s)), False))
        return out

    def loglikelihood_rolling(self, requests) -> List[float]:
        vals: List[float] = []
        for inst in requests:
            text = inst.args[0]
            vals.append(-1.0 * len(text) if text else float("-inf"))
        return vals

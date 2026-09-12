# Quality evaluation fixtures

`tools/quality.py` needs two corpora; neither is committed (a few MB of parquet). Regenerate:

```bash
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test*" --local-dir eval/wikitext
hf download cais/mmlu --repo-type dataset --include "all/test-*" --local-dir eval/mmlu
python3 -m venv eval/.venv && eval/.venv/bin/pip install pyarrow    # the image has no pyarrow
eval/.venv/bin/python - <<'PY'
import pyarrow.parquet as pq, json
t = pq.read_table('eval/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet').to_pydict()
open('eval/wikitext2-test.txt','w').write(''.join(t['text']))
m = pq.read_table('eval/mmlu/all/test-00000-of-00001.parquet').to_pydict()
json.dump([{'q':m['question'][i],'choices':list(m['choices'][i]),'answer':int(m['answer'][i]),
            'subject':m['subject'][i]} for i in range(len(m['question']))], open('eval/mmlu-test.json','w'))
PY
```

`eval/stdlib-code.txt` is six Python standard-library modules concatenated (dataclasses, argparse,
json/decoder, http/client, asyncio/tasks, statistics) as a code-domain corpus.

Run against a server, then compare:

```bash
DSV41_MODEL=DSv41Flash     python3 tools/quality.py ppl http://localhost:8004 eval/wikitext2-test.txt 73728 8192
DSV41_MODEL=DSv41Flash DSV41_MC_DUMP=eval/mc-base-1000.json python3 tools/quality.py mc http://localhost:8004 eval/mmlu-test.json 1000 0
# ... same on the other checkpoint, then
python3 tools/qcompare.py eval/base.txt eval/reap.txt eval/mc-base-1000.json eval/mc-reap-1000.json
```

## HumanEval+ / MBPP+

```bash
hf download evalplus/humanevalplus --repo-type dataset --local-dir eval/humanevalplus
hf download evalplus/mbppplus     --repo-type dataset --local-dir eval/mbppplus
eval/.venv/bin/python - <<'PY'
import pyarrow.parquet as pq, json, glob
for name in ("humanevalplus", "mbppplus"):
    t = pq.read_table(glob.glob(f"{name}/**/*.parquet", recursive=True)[0]).to_pydict()
    n = len(next(iter(t.values())))
    json.dump([{k: t[k][i] for k in t} for i in range(n)], open(f"eval/{name}.json", "w"))
PY
```

Generate, retry anything that hit the token cap, then score in the sandbox:

```bash
DSV41_MODEL=DSv41ReapFlash python3 tools/codeeval.py http://localhost:8005 humanevalplus eval/gen-x.json 0 8 4096
DSV41_MODEL=DSv41ReapFlash DSV41_RETRY=eval/gen-x.json python3 tools/codeeval.py http://localhost:8005 \
    humanevalplus eval/genr-x.json 0 8 16384
tools/coderun.sh eval/genr-x.json humanevalplus 30 > eval/score-x.json
```

`tools/coderun-all.sh` drives every server on the box in turn; `tools/qcode.py` builds the table.

**Validate the sandbox before trusting any model number**: score the datasets' own reference
solutions. Expect 378/378 on MBPP+ and 163/164 on HumanEval+ (HumanEval/32's reference solution
fails its own plus tests, so 163 is the ceiling).

```bash
tools/coderun.sh eval/canon-humanevalplus.json humanevalplus 30   # after building canon-*.json
```

`tools/coderun.sh` executes model output in a container with `--network none`, all capabilities
dropped, an unprivileged user, a tmpfs scratch dir and memory/CPU/pid caps. Never run generated
solutions on the host.

"""Corpus de 131072 tokens: bench_corpus.ids + copia rotada media vuelta."""
import pathlib
src = pathlib.Path("bench/fixtures/bench_corpus.ids").read_text().split()
out = src + src[len(src)//2:] + src[:len(src)//2]
dst = pathlib.Path("profiles/bench/bench_corpus_131072.ids")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(" ".join(out) + "\n")
print(len(out), "tokens ->", dst)

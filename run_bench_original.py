#!/usr/bin/env python3
"""
Original benchmark suite (recovered from Claude transcripts).
Matches the evaluation methodology used in the paper.
"""
import argparse
import random
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from configs import get_phase_config
from models import create_model


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')


def run_lambada(model, ds):
    correct, total = 0, 0
    with torch.no_grad():
        for item in ds:
            words = item['text'].rsplit(' ', 1)
            if len(words) < 2: continue
            ctx, last = words[0], ' ' + words[1]
            ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
            last_ids = tokenizer.encode(last, add_special_tokens=False)
            all_ids = (ctx_ids + last_ids)[-2048:]
            inp = torch.tensor([all_ids[:-1]], device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inp)
            pred = logits[0, len(all_ids)-len(last_ids)-1:].argmax(dim=-1)
            match = all(pred[i].item() == last_ids[i] for i in range(len(last_ids)) if len(all_ids)-len(last_ids)+i < len(all_ids))
            if match: correct += 1
            total += 1
    return correct / max(total, 1)


def run_niah(model, tests):
    correct, total = 0, 0
    with torch.no_grad():
        for seq, value_tokens in tests:
            inp = torch.tensor([seq], device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inp)
            if logits[0, -1].argmax().item() == value_tokens[0]: correct += 1
            total += 1
    return correct / max(total, 1)


def run_hellaswag(model, ds):
    correct, total = 0, 0
    with torch.no_grad():
        for item in ds:
            ctx = item['ctx']
            endings = item['endings']
            label = int(item['label'])
            scores = []
            for end in endings:
                ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
                end_ids = tokenizer.encode(end, add_special_tokens=False)
                all_ids = (ctx_ids + end_ids)[-2048:]
                inp = torch.tensor([all_ids[:-1]], device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(inp)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                cont_start = len(all_ids) - len(end_ids) - 1
                ll = sum(log_probs[0, cont_start + i, end_ids[i]].item() for i in range(len(end_ids)) if cont_start + i < log_probs.shape[1])
                scores.append(ll / max(len(end_ids), 1))
            if scores.index(max(scores)) == label: correct += 1
            total += 1
    return correct / max(total, 1)


def run_arc_easy(model, ds):
    correct, total = 0, 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
    with torch.no_grad():
        for item in ds:
            question = item['question']
            choices = item['choices']['text']
            label = label_map.get(item['answerKey'], -1)
            if label < 0 or label >= len(choices): continue
            scores = []
            for choice in choices:
                ctx_ids = tokenizer.encode(question, add_special_tokens=False)
                ch_ids = tokenizer.encode(' ' + choice, add_special_tokens=False)
                all_ids = (ctx_ids + ch_ids)[-2048:]
                inp = torch.tensor([all_ids[:-1]], device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(inp)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                cont_start = len(all_ids) - len(ch_ids) - 1
                ll = sum(log_probs[0, cont_start + i, ch_ids[i]].item() for i in range(len(ch_ids)) if cont_start + i < log_probs.shape[1])
                scores.append(ll / max(len(ch_ids), 1))
            if scores.index(max(scores)) == label: correct += 1
            total += 1
    return correct / max(total, 1)


def run_winogrande(model, ds):
    correct, total = 0, 0
    with torch.no_grad():
        for item in ds:
            sentence = item['sentence']
            opt1, opt2 = item['option1'], item['option2']
            label = int(item['answer']) - 1
            scores = []
            for opt in [opt1, opt2]:
                text = sentence.replace('_', opt)
                ids = tokenizer.encode(text, add_special_tokens=False)[-2048:]
                inp = torch.tensor([ids[:-1]], device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(inp)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                target = torch.tensor(ids[1:], device='cuda')
                ll = log_probs[0].gather(1, target.unsqueeze(1)).sum().item()
                scores.append(ll / len(ids))
            if scores.index(max(scores)) == label: correct += 1
            total += 1
    return correct / max(total, 1)


def build_niah_tests(n=200, seed=42, context_len=512):
    random.seed(seed)
    filler_text = 'The quick brown fox jumps over the lazy dog. '
    niah_pairs = [
        ('The secret code is', 'ALPHA'),
        ('The password is', 'BRAVO'),
        ('The answer is', 'CHARLIE'),
        ('The key number is', 'DELTA'),
        ('The magic word is', 'ECHO'),
    ]
    niah_tests = []
    for i in range(n):
        kp, v = random.choice(niah_pairs)
        needle = f'{kp} {v}.'
        ft = tokenizer.encode(filler_text * 50, add_special_tokens=False)
        nt = tokenizer.encode(needle, add_special_tokens=False)
        qt = tokenizer.encode(f' {kp}', add_special_tokens=False)
        avail = context_len - len(nt) - len(qt) - 10
        if avail < 10: continue
        pos = random.randint(5, avail // 2)
        seq = ft[:pos] + nt + ft[:avail - pos] + qt
        seq = seq[:context_len]
        vt = tokenizer.encode(f' {v}', add_special_tokens=False)
        niah_tests.append((seq, vt))
    return niah_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sisa')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--phase', type=int, default=2)
    ap.add_argument('--d-state', type=int, default=None)
    ap.add_argument('--d-ff', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--tasks', nargs='+', default=['lambada', 'niah', 'hellaswag', 'arc_easy', 'winogrande'])
    args = ap.parse_args()

    config = get_phase_config(args.phase, args.model)
    if args.d_state is not None:
        config.d_state = args.d_state
    if args.d_ff is not None:
        config.d_ff_reduced = args.d_ff

    print(f"\n===== {args.tag or args.model} =====")
    model = create_model(config)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to('cuda', dtype=torch.bfloat16).eval()

    print("Loading datasets...")
    datasets = {}
    if 'lambada' in args.tasks: datasets['lambada'] = load_dataset('lambada', split='test')
    if 'hellaswag' in args.tasks: datasets['hellaswag'] = load_dataset('Rowan/hellaswag', split='validation')
    if 'arc_easy' in args.tasks: datasets['arc_easy'] = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
    if 'winogrande' in args.tasks: datasets['winogrande'] = load_dataset('winogrande', 'winogrande_xl', split='validation')
    niah_tests = build_niah_tests() if 'niah' in args.tasks else []

    runners = {
        'lambada': lambda: run_lambada(model, datasets['lambada']),
        'niah': lambda: run_niah(model, niah_tests),
        'hellaswag': lambda: run_hellaswag(model, datasets['hellaswag']),
        'arc_easy': lambda: run_arc_easy(model, datasets['arc_easy']),
        'winogrande': lambda: run_winogrande(model, datasets['winogrande']),
    }

    results = {}
    for task in args.tasks:
        t0 = time.time()
        score = runners[task]()
        results[task] = score
        print(f"  {task}: {score:.4f}  ({time.time()-t0:.0f}s)")

    print(json.dumps({args.tag or args.model: results}, indent=2))


if __name__ == '__main__':
    main()

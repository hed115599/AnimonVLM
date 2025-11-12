# AnimonVLM
The official code for AnimonVLM. 

---

## 🏗️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/hed115599/AnimonVLM.git
cd AnimonVLM
```

### 2. Create a conda virtual environment
```bash
conda create -n AnimonVLM python=3.11 -y
conda activate AnimonVLM
```

### 3. Install dependencies
The recommend env is  CUDA 11.8 with pytorch  
```bash
pip install -r requirements.txt
```

---

## 📦 Data
This part will be released after the paper is accepted. 


### How to prepare your own data
Prepare your images in a folder and make a jsonl file as the label data. An example is given like:
```bash
{"image": "/home/hed/Public/DATAS/Bamapig/images/001124.png", "instances": [{"bbox": "pig<loc_203><loc_232><loc_535><loc_662>", "answer": "<loc_240><loc_550><loc_237><loc_530><loc_235><loc_501><loc_223><loc_453><loc_224><loc_442><loc_217><loc_420><loc_216><loc_405><loc_215><loc_383><loc_205><loc_338><loc_204><loc_297><loc_209><loc_272><loc_216><loc_248><loc_236><loc_235><loc_254><loc_235><loc_270><loc_239><loc_300><loc_275><loc_325><loc_306><loc_339><loc_317><loc_370><loc_314><loc_401><loc_307><loc_430><loc_316><loc_457><loc_333><loc_461><loc_334><loc_467><loc_332><loc_471><loc_321><loc_469><loc_315><loc_470><loc_308><loc_470><loc_300><loc_477><loc_303><loc_484><loc_316><loc_483><loc_349><loc_493><loc_372><loc_499><loc_396><loc_508><loc_418><loc_513><loc_450><loc_525><loc_477><loc_529><loc_487><loc_533><loc_483><loc_533><loc_490><loc_531><loc_502><loc_525><loc_516><loc_514><loc_534><loc_502><loc_529><loc_487><loc_512><loc_481><loc_517><loc_475><loc_517><loc_475><loc_507><loc_456><loc_499><loc_434><loc_509><loc_416><loc_509><loc_416><loc_529><loc_419><loc_546><loc_430><loc_560><loc_437><loc_570><loc_427><loc_571><loc_435><loc_581><loc_428><loc_585><loc_421><loc_583><loc_414><loc_576><loc_408><loc_561><loc_407><loc_579><loc_401><loc_550><loc_399><loc_529><loc_392><loc_509><loc_379><loc_520><loc_378><loc_556><loc_374><loc_590><loc_378><loc_620><loc_384><loc_642><loc_386><loc_649><loc_382><loc_656><loc_375><loc_649><loc_376><loc_661><loc_371><loc_659><loc_366><loc_650><loc_360><loc_619><loc_359><loc_637><loc_357><loc_618><loc_355><loc_605><loc_357><loc_582><loc_343><loc_507><loc_316><loc_512><loc_301><loc_509><loc_279><loc_491><loc_245><loc_432><loc_247><loc_469><loc_250><loc_499><loc_257><loc_533><loc_265><loc_554><loc_265><loc_564><loc_259><loc_557><loc_256><loc_558><loc_258><loc_572><loc_252><loc_570><loc_242><loc_543>"}]}
```

---

## 🚀 Training

We used jupyter notebook for training, it support multi-GPU training. 

**Example:**
```bash
notebook_launcher(
    train,
    args,
    num_processes=torch.cuda.device_count()
)
```

---

## 🔍 Inference

Once the model is trained, you can use the `infer_dataset` funtion to make predictions on new data.

**Example command:**
```bash
infer_dataset(peft_model.to(device), processor, train_dataset, [0,1,100,200, 500, 1000, -99,-3,-2], device)
```
The prediction results will be saved in the `results/` directory by default.

---

## 📚 Citation

We will give this part after the paper is accepted.

```bibtex
@misc{yourusername2025yourproject,
  title={Your Project Title},
  author={Your Name and Other Authors},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/your-repo-name}},
}
```

---

## 💬 Contact

If you have any questions, feedback, or suggestions, feel free to open an issue or reach out.

- **Your Name** – hed115599@gmail.com  
- **Project Link** – [https://github.com/hed115599/AnimonVLM](https://github.com/hed115599/AnimonVLM)  

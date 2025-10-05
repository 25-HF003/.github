## AI 기반 Proactive Deepfake Detection(선제적 딥페이크 탐지)


<div align="center">

<img width="1280" height="720" alt="썸네일 이미지" src="https://github.com/user-attachments/assets/598821b5-73a5-45fa-965e-958208b6d076" />

`#딥페이크 탐지` `#적대적 노이즈 삽입` `#워터마크 삽입/탐지` <br /> <br />
클릭 한 번으로 영상의 진실을 찾고 콘텐츠를 지키는 **통합 보안 솔루션** 'DeepTruth'입니다. 

</div>

---

### Repository
> Client: https://github.com/25-HF003/front-end <br />
> Spring Boot server: https://github.com/25-HF003/back-end <br/>
> Deepfake server: https://github.com/25-HF003/deepFake <br />
> Adversarial-Noise server: https://github.com/25-HF003/Adversarial-Noise <br />
> Watermark server: https://github.com/25-HF003/Watermark <br />

---

## **💡1. 프로젝트 개요**


**1-1. 프로젝트 소개**
- 프로젝트 명 : DeepTruth
- 프로젝트 정의 : 불법적인 딥페이크 탐지 및 생성 방지를 위한 AI 기반 영상 분석 솔루션

**1-2. 개발 배경 및 필요성**
- 최근 딥페이크 기술의 악용 사례가 늘어 사회적 문제로 대두되고 있습니다. 정치인 가짜 연설 영상과 유명인 합성 음란물 유포는 물론, AI의 무단 학습으로 원작자의 저작권 침해 문제까지 발생하고 있습니다. 이에 본 프로젝트는 AI 기반 딥페이크 탐지 시스템을 개발합니다. 또한 적대적 노이즈와 보이지 않는 워터마크 기술을 적용하여 딥페이크 탐지 및 예방 기능을 강화하는 것을 목표로 합니다.

**1-3. 프로젝트 특장점**
* 통합형 보안 시스템: 딥페이크 탐지, 적대적 노이즈, 비가시적 워터마크 기술을 결합한 통합형 보안 솔루션
* 비가시적 워터마크 기술: 육안으로는 보이지 않지만 AI가 인식 가능한 워터마크로, 콘텐츠 소유권 보호 및 변조 추적에 활용됨
* AI 학습 방해 기술: 적대적 노이즈를 활용하여 AI의 무단 학습을 방해하고 지적 자산을 보호하는 기능
* 사용자 맞춤형 이원화 기능: 일반/정밀 모드 및 자동/정밀 모드를 제공하여 사용자의 목적과 필요에 맞춰 최적화된 기능을 선택할 수 있도록 설계된 점
* 확장성 및 유연성: 사용자가 직접 설정을 조정하여 다양한 환경과 요구 사항에 유연하게 대응하는 실용적 구조

**1-4. 주요 기능**
- 딥페이크 탐지 : 기본모드/정밀모드 선택 가능, 딥페이크 탐지 결과는 위험확률(%)과 함께 의심영역을 보여주고 탐지된 장면의 이미지를 함께 제공
- 적대적 노이즈 삽입: 자동모드/정밀모드 선택 가능, 적대적 노이즈 삽입 이미지와 함께 공격 성공여부, 분류 변화, 신뢰도 변화, 적대적 노이즈 강도 정보 제공
- 워터마크 삽입:  파일을 받아 워터마크를 삽입/탐지 가능

**1-5. 기대 효과 및 활용 분야**
- 기대효과: 사회·정치적 안정성 확보, 창작자 지적 재산권 보호, 차세대 AI 보안 기술 표준 정립
- 활용방안: 사법 절차 법적 대응 지원, 언론·미디어의 보도 영상 무결성 검증, 디지털 작품 저작권 보호

**1-6. 기술 스택**
- 프론트엔드 : <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=white"> <img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=Next.js&logoColor=white"/> <img src="https://img.shields.io/badge/typescript-%23007ACC.svg?style=flat-square&logo=typescript&logoColor=white"> <img src="https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat-square&logo=tailwind-css&logoColor=white">
- 백엔드 : <img src="https://img.shields.io/badge/java-007396?style=flat-square&logo=OpenJDK&logoColor=white"> <img src="https://img.shields.io/badge/Spring Boot-6DB33F?style=flat-square&logo=springboot&logoColor=black"/>
- AI/ML : <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=Flask&logoColor=white">
- 데이터베이스 : <img src="https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=MySQL&logoColor=white"> 
- 클라우드 :  <img src="https://img.shields.io/badge/Amazon AWS-FF9900?style=flat-square&logo=amazonec2&logoColor=white"/>
- 배포 및 관리 : <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white"> 

---


## **💡2. 팀원 소개**
| <img width="80" height="100" src="https://avatars.githubusercontent.com/u/90364648?v=4" alt="강수정"> | <img width="80" height="100" alt="김보민" src="https://avatars.githubusercontent.com/u/101878770?v=4" > | <img width="80" height="100" src="https://avatars.githubusercontent.com/u/123048615?v=4" width=90px alt="서지혜"/>| <img width="80" height="100" alt="여강휘" src="https://avatars.githubusercontent.com/u/101783655?v=4" > | 
|:---:|:---:|:---:|:---:|
| **강수정** | **김보민** | **서지혜(팀장)** | **여강휘** | 
| [@kangsujung](https://github.com/kangsujung) | [@fsdffds](https://github.com/fsdffds)  | [@Jihye0623](https://github.com/jihye0623) | [@YO1231](https://github.com/YO1231) |
| • FrontEnd <br> • AI | • FrontEnd <br> • AI | • BackEnd <br> • AI |• BackEnd <br> • AI |

---
## **💡3. 시스템 구성도**
- 서비스 구성도
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/78a8aae8-a3c3-418e-bd3e-0685a59d9413" />

- 엔티티 관계도
<img width="500" height="500" alt="스크린샷(8)" src="https://github.com/user-attachments/assets/779c8c02-8341-45d1-a807-16c7cc6812d2" />


---
## **💡4. 작품 소개영상**
[![Video Label](http://img.youtube.com/vi/6XYoLJTWHXI/0.jpg)](https://youtu.be/6XYoLJTWHXI?si=1JbJn2mcm2SDC6fU)

---
## **💡5. 핵심 소스코드**

### **:boom:딥페이크 탐지 기능**
#### 1) DNN으로 얼굴 검출을 시도하고, 실패하면 이미지를 리사이즈하여 재시도하거나 dlib 탐지기로 전환하는 등 여러 단계의 폴백(fallback)을 통해 안정적으로 얼굴을 찾아내는 함수입니다.

```python
def robust_detect(frame, *, detector="dnn", dnn_conf=0.30, resize_long=720, max_boxes=5):
    H, W = frame.shape[:2]
    # 1차 DNN
    try:
        bboxes = detect_face_bboxes(frame, detector="dnn", dnn_conf=dnn_conf, max_boxes=max_boxes)
    except Exception as e:
        print("[ERR] dnn first pass:", repr(e), flush=True); 
        bboxes = []

    # 리사이즈 후 재시도
    if not bboxes:
        long_side = resize_long or 720
        scale = float(long_side) / float(max(H, W))
        fr = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        try:
            b2 = detect_face_bboxes(fr, detector="dnn", dnn_conf=0.25, max_boxes=max_boxes)
        except Exception as e:
            print("[ERR] dnn resized pass:", repr(e), flush=True); 
            b2 = []
        if b2:
            inv = (1.0/scale) if scale>0 else 1.0
            bboxes = [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv), conf) for (x1,y1,x2,y2,conf) in b2]
    
    # 폴백
    if not bboxes and detector != "dlib":
        try:
            fb = detect_face_bboxes(frame, detector="dlib", max_boxes=max_boxes) or []
            if fb:
                print("[DBG] fallback dlib hit", flush=True)
            bboxes = fb
        except Exception as e:
            print("[ERR] dlib fallback:", repr(e), flush=True); 
    return bboxes
```

#### 2) 동영상에서 프레임을 균등하게 샘플링하여 딥페이크 여부를 추론하고, 지수 이동 평균(EMA)으로 신뢰도 점수를 보정하여 최종 결과를 계산하는 기능을 수행합니다.

```python
# 균등 샘플링
step = max(1, num_frames // max(1, sample_count))
target_indices = set([min(i*step, num_frames-1) for i in range(max(1, sample_count))])

ema = None
per_frame_conf = []
raw_conf_for_vote = []
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    if frame_idx in target_indices:
        # (정밀/기본 모드에 따라 얼굴 검출/전처리/추론)
        # ...
        # 추론 후 EMA 적용
        if ema is None:
            ema = conf_i  # 또는 conf
        else:
            ema = 0.5*conf_i + 0.5*ema
        conf_s = float(ema)

        raw_conf_for_vote.append((conf_i, q, l))
        per_frame_conf.append(conf_s)
        results.append({'pred': 1 if conf_s>=0.5 else 0, 'confidence': conf_s})
        # ...
    frame_idx += 1

```
<br/>

### **:boom:적대적 노이즈 삽입 기능**
#### 자동(신뢰도 기반) 또는 정밀(사용자 레벨 선택) 모드로 FGSM 공격 및 Gaussian Blur 후처리를 수행하는 함수입니다.

```python
def fgsm_attack_with_blur(image_tensor, base_epsilon=0.015, base_sigma=0.4, mode='auto', level=2):
    image_tensor = image_tensor.clone().unsqueeze(0).requires_grad_(True)
    
    result = classify_with_art_model(image_tensor)
    if result[0] is None:  # 분류 실패 시
        raise ValueError("원본 이미지 분류에 실패했습니다.")
    original_class, conf, original_pred = result
    
    # 모드별 epsilon 결정
    if mode == 'precision':
        # 정밀 모드: 자동 모드의 각 단계와 동일한 epsilon 사용
        epsilon_levels = {
            1: base_epsilon,        # 기본 (1.0배)
            2: base_epsilon * 1.5,  # 중간 (1.5배)
            3: base_epsilon * 2.5,  # 강함 (2.5배)
            4: base_epsilon * 4.0   # 매우 강함 (4.0배)
        }
        eps = epsilon_levels.get(level, base_epsilon)
        sigma = base_sigma  # 고정
        auto_reason = None
        
    else:  # mode == 'auto'
        # 자동 모드: 신뢰도 기반 조정 (기존 로직)
        if conf > 0.99:
            eps = base_epsilon * 4.0
            sigma = base_sigma * 0.3
            auto_reason = "very_high_confidence"
        elif conf > 0.95:
            eps = base_epsilon * 2.5
            sigma = base_sigma * 0.5
            auto_reason = "high_confidence"
        elif conf > 0.9:
            eps = base_epsilon * 1.5
            sigma = base_sigma
            auto_reason = "medium_confidence"
        else:
            eps = base_epsilon
            sigma = base_sigma
            auto_reason = "low_confidence"
    
    # FGSM 공격
    try:
        # gradient 계산
        image_pil = transforms.ToPILImage()(image_tensor.squeeze().clamp(0, 1))
        inputs = art_processor(images=image_pil, return_tensors="pt")
        inputs['pixel_values'].requires_grad_(True)
        
        outputs = art_model(**inputs)
        target = torch.tensor([original_pred])
        loss = F.cross_entropy(outputs.logits, target)
        
        # Gradient 기반 perturbation
        loss.backward()
        
        if inputs['pixel_values'].grad is not None:
            perturbation = eps * inputs['pixel_values'].grad.sign()
            # 크기 맞춤
            if perturbation.shape != image_tensor.shape:
                perturbation = F.interpolate(perturbation, size=image_tensor.shape[2:], mode='bilinear')
            adv_image = image_tensor + perturbation
            print("[DEBUG] Gradient 기반 FGSM 적용")
        else:
            raise Exception("Gradient 계산 실패")
            
    except Exception as e:
        print(f"[WARN] 예술 모델 gradient 실패, fallback 사용: {e}")
        # 기존 방식으로 fallback
        perturbation = eps * torch.randn_like(image_tensor)
        adv_image = image_tensor + perturbation
    
    adv_image = torch.clamp(adv_image, 0, 1)
    
    # 가우시안 블러
    adv_np = adv_image.squeeze(0).detach().cpu().numpy()
    adv_blur_np = np.stack([gaussian_filter(c, sigma=sigma) for c in adv_np])
    adv_blur = torch.from_numpy(adv_blur_np).unsqueeze(0)
```
<br/>

### **:boom:워터마크 삽입 기능**
```python
@app.route('/watermark-insert', methods=['POST'])
def watermarkInsert():
    # 1. 이미지와 메시지 받기
    image_file = request.files.get('image')
    message = request.form.get('message', 'ETNL')
    assert len(message) <= 4, "메시지는 4자 이하만 가능"
    if not image_file or not message:
        return jsonify({"error": "image, message 둘 다 필요합니다."}), 400
        
    # 2. 이미지 로드 및 전처리
    image = Image.open(image_file.stream).convert("RGB")
    img_pt = default_transform(image).unsqueeze(0).to(device)
    
    # 3. 메시지 전처리
    wm_bits = ''.join(f"{ord(c):08b}" for c in message)
    wm_bits = wm_bits.ljust(32, '0')[:32]
    wm_msg = torch.tensor([[int(bit) for bit in wm_bits]], dtype=torch.float32).to(device)
    
    # 3. 워터마크 삽입
    outputs = wam.embed(img_pt, wm_msg)
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.5)
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
    
    # 4. 이미지 후처리 
    out_img = unnormalize_img(img_w).squeeze(0).detach().clamp_(0, 1)  # 1. 정규화 해제 + 값 범위 제한 (0~1)
    out_img_np = out_img.permute(1, 2, 0).cpu().numpy()                # 2. CPU로 이동 후 numpy 변환 (HWC 형태)
    out_img_np = (out_img_np * 255).round().astype('uint8')            # 3. 0~255 범위로 변환 (소수점 처리 개선)
    out_img_pil = Image.fromarray(out_img_np)
```
<br/>

### **:boom:워터마크 탐지 기능**
```python
@app.route('/watermark-detection', methods=['POST'])
def watermarkDetection():
    # 1. 이미지 수신 및 기본 정보 추출
    image_file = request.files.get('image')
    message = request.form.get('message', '')
    if not image_file or not message:
        return jsonify({"error": "image, message 둘 다 필요합니다."}), 400
        
    # 2. 이미지 전처리
    image = Image.open(image_file.stream).convert("RGB")
    img_pt = default_transform(image).unsqueeze(0).to(device)
    
    # 3. 워터마크 탐지 (모델 추론)
    with torch.no_grad():
        detect_outputs = wam.detect(img_pt)
        preds = detect_outputs['preds']      # shape: [B, 1+nbits, H, W]
        mask_preds = preds[:, 0:1, :, :]     # 예측된 마스크
        bit_preds = preds[:, 1:, :, :]       # 예측된 메시지 비트
        
    # 4. 예측된 비트로부터 메시지 추출
    pred_message = msg_predict_inference(bit_preds, mask_preds)
    pred_message_float = pred_message.float()  # float32로 변환
    
    # 5. 원본 메시지 텐서 변환
    wm_bits = ''.join(f"{ord(c):08b}" for c in message.ljust(4, '\x00'))[:32]
    wm_tensor = torch.tensor([int(b) for b in wm_bits], dtype=torch.float32).to(device)
    
    # 6. 비트 정확도 계산
    bit_acc = (pred_message_float == wm_tensor.unsqueeze(0)).float().mean().item()
    bit_acc_pct = round(bit_acc * 100, 1)
```

## AI 기반 Proactive Deepfake Detection(선제적 딥페이크 탐지)


<div align="center">
	

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
- 기대효과: 사회·정치적 안정성 확보, 창작자 지적 재산권 보호, 차세대 AI 보안 기술 표준 정립.
- 활용방안 디지털 미디어 신뢰성·안전성 강화, 사법 절차 법적 대응 지원

**1-6. 기술 스택**
- 프론트엔드 : <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=white"> <img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=Next.js&logoColor=white"/> <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=CSS3&logoColor=white">
- 백엔드 : <img src="https://img.shields.io/badge/java-007396?style=flat-square&logo=OpenJDK&logoColor=white"> <img src="https://img.shields.io/badge/Spring Boot-6DB33F?style=flat-square&logo=springboot&logoColor=black"/>
- AI/ML : <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=Flask&logoColor=white">
- 데이터베이스 : <img src="https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=MySQL&logoColor=white"> 
- 클라우드 : <img src="https://img.shields.io/badge/Amazon AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white"/>
- 배포 및 관리 : <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=Kubernetes&logoColor=white"> 

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

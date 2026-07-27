# 프로젝트: 비시각적 멀티모달 물체 인식 시스템

## 연구 목표
시각 없이 촉각 + 접촉 진동 + 후각으로 물체를 인식하는 시스템.
해석 가능한 피처 기반 판단이 핵심. 블랙박스 ML이 아님.

## 센서 구성
- WowSkin: 촉각 (200~500 Hz, Teensy ADC)
- Knowles BU x2: 접촉 진동 (16 kHz, Teensy ADC)
- ENS160: TVOC/eCO₂ (1~2 Hz, I2C 0x53)
- BME280: 온습도/기압 (1 Hz, I2C 0x77)
- BME688: 히터 프로필 gas fingerprint (~10.8초/사이클, I2C 0x76)

## MCU
- Teensy 4.1 (USB Serial → 노트북)
- Arduino IDE로 펌웨어 업로드

## 데이터 구조
data/raw/obj{번호}_{이름}/trial_{번호}/
├── tactile.csv (time_ms, ch1, ch2, ...)
├── acoustic.csv (time_ms, bu1_raw, bu2_raw)
├── gas_ens160.csv (time_ms, tvoc_ppb, eco2_ppm, aqi)
├── gas_bme688.csv (time_ms, heater_step, gas_resistance_ohm, ...)
├── environment.csv (time_ms, temp, hum, pressure)
└── meta.json

## Python 환경
conda activate nonvisual
Python 3.11
주요 패키지: pyserial, numpy, scipy, pandas, matplotlib, scikit-learn, librosa

## 코드 스타일
- 함수에 docstring 필수
- 타입 힌트 사용
- 변수명: snake_case
- 파일 인코딩: UTF-8
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, convertInchesToTwip, TableLayoutType,
} = require("docx");
const fs = require("fs");

const FONT = "맑은 고딕";
const MONO = "D2Coding";           // 없으면 시스템 대체 (Consolas 계열)
const MONO_FB = "Consolas";
const CW = 9638;                    // A4 - 좌우 여백 각 2cm

const C = {
  ink: "1A1D21", ink2: "44505A", ink3: "77848D",
  rule: "C4CDD4", head: "E8EDF0", accent: "0F6E33", warn: "9A5B0B",
};

/* ---------- helpers ---------- */
const P = (text, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: 300 },
  alignment: o.align,
  indent: o.indent,
  children: [new TextRun({
    text, font: o.mono ? MONO : FONT, size: o.size ?? 20,
    bold: o.bold, italics: o.italic, color: o.color ?? C.ink,
  })],
});

const Rich = (runs, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: 300 },
  indent: o.indent,
  children: runs.map(r => new TextRun({
    text: r.t, font: r.mono ? MONO : FONT, size: r.size ?? 20,
    bold: r.b, italics: r.i, color: r.c ?? C.ink,
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.rule, space: 6 } },
  children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: C.ink })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 260, after: 140 },
  children: [new TextRun({ text, font: FONT, size: 23, bold: true, color: C.ink })],
});

const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 200, after: 110 },
  children: [new TextRun({ text, font: FONT, size: 21, bold: true, color: C.ink2 })],
});

const Bul = (text, lvl = 0) => new Paragraph({
  numbering: { reference: "bul", level: lvl },
  spacing: { after: 90, line: 300 },
  children: [new TextRun({ text, font: FONT, size: 20, color: C.ink })],
});

const Code = (lines) => lines.map((l, i) => new Paragraph({
  spacing: { before: i === 0 ? 90 : 0, after: i === lines.length - 1 ? 150 : 0, line: 250 },
  shading: { type: ShadingType.CLEAR, fill: "F2F5F7" },
  indent: { left: 220, right: 220 },
  children: [new TextRun({ text: l || " ", font: MONO, size: 17, color: C.ink })],
}));

const Note = (text) => new Paragraph({
  spacing: { before: 120, after: 160, line: 300 },
  indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 14, color: C.warn, space: 10 } },
  children: [new TextRun({ text, font: FONT, size: 19, color: C.ink2 })],
});

function Tbl(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CW / total;
  const w = widths.map(x => Math.round(x * scale));
  const cell = (txt, i, isHead, mono, color) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: C.head } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: String(txt).split(" ").map((line, k) => new Paragraph({
      spacing: { after: 0, line: 260 },
      children: [new TextRun({
        text: line, font: mono ? MONO : FONT, size: isHead ? 18 : 18,
        bold: isHead, color: color ?? (isHead ? C.ink : C.ink),
      })],
    })),
  });
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: w,
    layout: TableLayoutType.FIXED,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: C.rule },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: C.rule },
      left: { style: BorderStyle.SINGLE, size: 6, color: C.rule },
      right: { style: BorderStyle.SINGLE, size: 6, color: C.rule },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: C.rule },
      insideVertical: { style: BorderStyle.SINGLE, size: 4, color: C.rule },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, i, true)),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((cv, i) => {
          const isObj = cv && typeof cv === "object" && !Array.isArray(cv);
          return cell(isObj ? cv.t : cv, i, false, isObj ? cv.mono : false,
                      isObj ? cv.c : undefined);
        }),
      })),
    ],
  });
}
const Gap = (h = 120) => new Paragraph({ spacing: { after: h }, children: [] });
const M = (t) => ({ t, mono: true });
// 표 셀 안 줄바꿈 구분자 (Tbl 이 이 문자로 문단을 나눈다)
const SEP = "\u2028";

/* ================= 본문 ================= */
const body = [];

/* ---------- 표지 ---------- */
body.push(
  new Paragraph({ spacing: { before: 1800, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "레이저 그리드 기반 현장 품질검측 장비",
      font: FONT, size: 26, color: C.ink2 })] }),
  new Paragraph({ spacing: { before: 160, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "영역별 품질검측 파이프라인",
      font: FONT, size: 48, bold: true, color: C.ink })] }),
  new Paragraph({ spacing: { before: 100, after: 500 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "구조 · 파라미터 · 입력 데이터 정리",
      font: FONT, size: 30, color: C.ink2 })] }),
  new Paragraph({ spacing: { before: 0, after: 0 }, alignment: AlignmentType.CENTER,
    border: { top: { style: BorderStyle.SINGLE, size: 8, color: C.rule, space: 14 } },
    children: [new TextRun({ text: " ", font: FONT, size: 20 })] }),
  new Paragraph({ spacing: { before: 300, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "광운대학교 건설시스템공학 연구실",
      font: FONT, size: 22, color: C.ink })] }),
  new Paragraph({ spacing: { after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "저장소 znlsl10-rgb/capcut  ·  브랜치 claude/laser-image-segmentation-smoothness-yklpvp",
      font: MONO, size: 16, color: C.ink3 })] }),
  new Paragraph({ spacing: { after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "2026-08-25 기준  ·  Python 17개 파일, 8,600줄",
      font: FONT, size: 18, color: C.ink3 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

/* ================= 1. 개요 ================= */
body.push(H1("1. 개요"));

body.push(H2("1.1 목적"));
body.push(P("본 문서는 레이저 그리드 품질검측 장비의 소프트웨어를 「스테이션 단위 검측」에서 「영역 단위 검측」으로 전환한 작업을 정리한 것이다. 현장 사진 한 장에 벽·바닥·동바리·철근이 함께 들어와도, 영역을 나누어 각 부재에 맞는 검측식을 적용하는 것이 목표다."));

body.push(H2("1.2 기존 구조의 한계"));
body.push(P("기존 파이프라인은 검측 대상과 검측 항목을 설정 파일(STATIONS 딕셔너리)에 미리 적어두고, 화면 전체를 하나의 평면으로 적합했다. 즉 「무엇을 잴 것인가」가 인지 결과가 아니라 설정값이었다. 이 구조에서는 다음이 성립하지 않는다."));
body.push(Bul("한 장에 벽과 바닥이 같이 들어오면 두 면 사이 어딘가로 평면이 잡힌다."));
body.push(Bul("동바리·철근은 1D 선형 부재라 평면 적합 자체가 성립하지 않는다."));
body.push(Bul("수직도와 수평도가 서로 다른 좌표축(n_y / n_z)을 읽고 있어, 장비를 각 면에 정면으로 겨눈다는 전제에서만 옳다."));

body.push(H2("1.3 변경의 핵심"));
body.push(P("구조 변경의 실체는 「무엇을 잴지 정하는 신호」의 출처가 설정 파일에서 이미지로 옮겨온 것 하나이며, 나머지는 전부 그 결과다."));
body.push(Gap(60));
body.push(Tbl(
  ["구분", "기존 — 스테이션 단위", "지금 — 영역 단위"],
  [
    ["검측 대상 결정", "설정 파일(STATIONS)에 사전 기재", "세그멘테이션 결과에서 결정"],
    ["평면 가정", "화면 전체 = 단일 평면", "영역별 개별 적합"],
    ["기준 좌표계", "조사기 좌표계 축 고정", "중력벡터 ĝ 기준 통합"],
    ["부재 종류", "면(평면)만", "면 + 선형 부재(축)"],
    ["출력", "검측값 1개", "영역별 검측값 N개 + 판정"],
  ],
  [1700, 3900, 4038]));

/* ================= 2. 전체 구조 ================= */
body.push(H1("2. 전체 구조 및 데이터 흐름"));

body.push(H2("2.1 처리 흐름"));
body.push(P("1회 촬영은 레이저 ON/OFF 두 프레임으로 갈라진다. 레이저가 켜진 프레임은 격자선이 화면을 덮어 세그멘테이션을 방해하므로, 영역 분할은 OFF 프레임이 맡고 선검출만 ON 프레임을 쓴다. 두 프레임은 수십 µs 간격이라 마스크가 픽셀 단위로 그대로 정합되며, 이 방식은 차영상 모드를 위해 이미 하드웨어 사양에 있으므로 추가 촬영 비용이 없다."));
body.push(...Code([
  "1회 촬영  =  (rgb_off, rgb_on, IMU ĝ, 캘리브레이션)",
  "",
  "  ├─[A ] 선검출        rgb_on  → {lid: [(u,v)…]}",
  "  ├─[C ] 영역분할      rgb_off → label_map (H×W 클래스 id)",
  "  ├─[eq1] 삼각측량             → (X,Y,Z) 조사기 좌표계",
  "  ├─[eq5] 영역 할당            → 마스크 침식 · 깊이 불연속 제거",
  "  │                              의미×기하 융합 · 병합영역 재분할",
  "  └─ 영역별 검측",
  "       면 부재   (벽·바닥·거푸집·조적·슬래브)",
  "           eq2 평면 적합(TLS) → eq3 수직도/수평도",
  "                              → eq4 요철위치 + eq6 KCS 직선자",
  "       선형 부재 (동바리·기둥·철근)",
  "           eq2 축 적합(RANSAC) → eq3 축 수직도   [평활도 없음]",
  "",
  "       전 영역 → eq5 불확실도 → 합격 / 기준초과 /",
  "                                 측정불가 / 판정보류(분해능)",
]));
body.push(Note("벽과 바닥은 서로 다른 검측이 아니라 같은 식에 중력 ĝ만 다르게 들어가는 것이다. 실제로 갈라지는 지점은 「면이냐 축이냐」이며, 이는 부재의 기하 형상에 따른 구분이다."));

body.push(H2("2.2 파일 구성"));
body.push(P("계산 로직은 전부 Isaac Sim에 의존하지 않는 모듈에 두었다. 렌더링 없이도 같은 코드로 정답 대조가 가능하며, Isaac 쪽은 씬 구성과 촬영·렌더만 담당한다."));
body.push(Gap(60));
body.push(Tbl(
  ["단계", "파일", "역할"],
  [
    ["씬·촬영 (Isaac)", M("1-1_build_inspection_lab_realistic.py"),
     "USD 씬 + Semantics 라벨 부여. Replicator 어노테이터가 화소별 정답 마스크를 제공. StationC_Mixed = 벽+바닥+동바리 3본+철근 2본"],
    ["", M("inspection.py"), "촬영 → 선검출 → 시맨틱 마스크 → 영역별 검측. inspect=\"auto\"면 검측 종류를 세그멘테이션이 결정"],
    ["인지", M("A_선검출.py"), "20×20 격자 선검출. multi_surface로 단일 평면 가정 해제"],
    ["", M("C_영역분할.py"), "gt(Isaac Semantics)·geom(다중평면 RANSAC) 구현. sam·vlm은 인터페이스만"],
    ["검측식", M("calibration.py"), "캘리브레이션 단일 출처. 사양에서 f·발산각을 유도하고 값마다 출처 등급(spec/design/assumed)을 표기"],
    ["", M("eq1_triangulation.py"), "능동 삼각측량"],
    ["", M("eq2_plane_fit.py"), "TLS 평면적합, PCA·RANSAC 축적합, 면내 좌표계"],
    ["", M("eq3_orientation.py"), "중력 기준 수직·수평·축 수직도, KCS 판정"],
    ["", M("eq4_flatness_line.py"), "요철 위치·깊이 (면내 격자)"],
    ["", M("eq5_region_assign.py"), "영역 할당, 경계 정제, 의미×기하 융합, 불확실도"],
    ["", M("eq6_straightedge.py"), "KCS 직선자 판정 + 분해능 편향 진단"],
    ["통합·출력", M("pipeline_region.py"), "영역별 검측 오케스트레이션 (Isaac 비의존)"],
    ["", M("report.py"), "검측 조서(JSON)·요약표·영역 오버레이"],
    ["검증", M("synth_scene.py"), "벽+바닥+동바리 해석적 합성 씬 (카메라 가림 포함)"],
    ["", M("experiment_segmentation.py"), "세그멘테이션 오차 분해 실험"],
    ["", M("experiment.py"), "기선 × 측정거리 sweep (Isaac)"],
    ["", M("inspect_png.py"), "렌더/촬영 이미지 한 장 검측 + 사양 대조(--check)"],
    ["", M("experiment_spec.py"), "사양 프로파일 정확도 비교"],
    ["", M("tests/test_regression.py"), "9군 회귀 검증 (프로파일 3종)"],
  ],
  [1200, 3100, 5338]));

/* ================= 3. 알고리즘 ================= */
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("3. 품질검측 알고리즘"));

body.push(H2("3.1 식 ① 능동 삼각측량"));
body.push(...Code([
  "Z(i,j) = f·b / [ f·tan(α_i) − (u(i,j) − c_x) ]",
  "X(i,j) = (u − c_x)·Z/f + b",
  "Y(i,j) = (v − c_y)·Z/f",
]));
body.push(P("좌표계는 조사기 기준이며 X는 우측, Y는 하단(이미지 v 증가 방향), Z는 전방(작업거리)이다. 기선 b가 X축 방향이므로 시차는 u 좌표에만 실린다. 따라서 깊이를 풀려면 그 점의 발사각 α를 알아야 하고, α가 선마다 고정된 V선만 단독으로 삼각측량이 가능하다. H선은 β만 고정이고 α가 점마다 다르며, v 좌표는 이미 아는 tan(β)를 되풀이할 뿐이라 깊이 정보를 담지 않는다. H선은 V선과의 교점에서만 α를 회복할 수 있고, 그 교점은 이미 V선 조밀 샘플에 포함된다."));

body.push(H2("3.2 식 ②③ 자세 — 중력 기준 통합"));
body.push(P("기존 v1은 조사기 좌표계의 특정 축을 직접 읽었다. 벽은 arcsin(|n_y|), 바닥은 arccos(|n_z|)로 서로 다른 축을 썼는데, 이는 장비를 각 검측면에 정면으로 겨눈다는 전제에서만 성립한다. 한 장에 벽과 바닥이 동시에 들어오면 두 면에 대해 동시에 정면일 수 없으므로 이 전제가 깨진다."));
body.push(P("축을 고정하지 않고 중력벡터 ĝ를 명시적으로 받도록 일반화하였다."));
body.push(...Code([
  "벽    (면)  θ_vert  = arcsin |n̂ · ĝ|      완전 수직 → 0°",
  "바닥  (면)  θ_horiz = arccos |n̂ · ĝ|      완전 수평 → 0°",
  "동바리(축)  θ_vert  = arccos |d̂ · ĝ|      완전 수직 → 0°",
]));
body.push(P("기존 식은 이 일반식의 특수해다. ĝ=(0,1,0)이면 arcsin|n̂·ĝ| = arcsin|n_y|이고, ĝ=(0,0,1)이면 arccos|n̂·ĝ| = arccos|n_z|가 되어 v1과 수치적으로 동일하다. 따라서 기존 검증값(수직도 오차 0.0008°, 수평도 오차 0°)이 그대로 재현된다."));
body.push(Note("작업 중 확인: v1의 수평도 자체검증이 88°(참값 2°)로 이미 실패 상태였다. 벽과 바닥에서 서로 다른 축을 읽던 좌표계 가정이 원인이며, 중력 기준으로 통합하자 구조적으로 해소되었다."));
body.push(P("선형 부재(동바리·기둥·철근)는 평면이 아니므로 법선이 아니라 축 방향 d̂를 쓴다. 축은 PCA 1주성분으로 구하되, 얇은 부재는 마스크 실루엣에서 반드시 오염되므로 RANSAC 기반 robust 적합을 기본으로 한다. 또한 세그멘테이션이 부재의 3D 실제 길이를 주므로, 각도뿐 아니라 KCS가 규정하는 mm 편차(편차 = h·tanθ) 판정이 가능해진다."));

body.push(H2("3.3 식 ④ 요철 검출"));
body.push(P("영역 평면의 접선 기저로 좌표변환한 뒤 면내 (u,v) 격자에서 국소 median으로 노이즈를 억제하고, 임계 초과 후보를 DBSCAN으로 공간 검증하여 요철 클러스터를 확정한다. 요철 깊이는 평활값이 아니라 정점 근방 원시 잔차의 median에서 산출한다. 평활은 검출용 노이즈 억제 수단이며, 창이 요철보다 넓으면 깊이를 그만큼 깎기 때문이다."));

body.push(H2("3.4 식 ⑤ 영역 할당 및 불확실도"));
body.push(P("세그멘테이션이 준 픽셀 라벨맵과 삼각측량이 준 3D 점을 결합해 검측 단위인 영역을 만든다. 다음 네 가지를 수행한다."));
body.push(Bul("마스크 침식 — 경계 근처 격자점은 마스크 부정확·서브픽셀 혼합·실루엣 삼각측량 오차가 겹치므로 제외한다."));
body.push(Bul("깊이 불연속 제거 — 선을 따라 Z가 급변하는 지점의 양옆 점을 제거한다."));
body.push(Bul("의미×기하 융합 — 벽/바닥 구분과 면/선형 구분은 기하가 우선하고, 동바리·기둥·철근 구분과 벽·거푸집·조적 구분은 의미(라벨)가 우선한다. 기하가 구분할 수 없는 축은 라벨을 신뢰하는 비대칭 규칙이다."));
body.push(Bul("병합 영역 재분할 — 세그멘테이션이 두 부재를 같은 라벨로 묶으면 영역 적합이 두 면에 걸쳐 소수쪽 부재가 사라진다. 영역 내부 기하 부정합을 검사해 되쪼갠다."));
body.push(Gap(60));
body.push(P("불확실도는 다음과 같이 산출한다."));
body.push(...Code([
  "σ_Z = σ_u · Z² / (f · b)              깊이 방향",
  "σ_n = σ_Z / |cos φ|                   법선 방향 (φ = 입사각)",
]));
body.push(P("같은 사진 안에서도 정면인 벽은 φ≈0이라 σ_n≈σ_Z이지만, 비스듬히 들어온 바닥은 φ가 커져 σ_n이 크게 증폭된다. 평활도는 법선 방향 오차가 곧 측정치이므로 이 증폭을 반영해야 판정이 정직해진다. σ_n이 목표(±2mm)를 넘으면 값은 참고로 남기되 판정은 하지 않는다."));

body.push(H2("3.5 식 ⑥ KCS 직선자 평활도"));
body.push(P("기존 평활도는 전역 평면 잔차였으나 KCS가 규정하는 판정값은 그것이 아니다. KCS 14 20 10은 「3m 직선자에 의한 처짐량」, KCS 41 46 00은 1m당 10mm로, 현장 검사원이 자를 얹고 자와 표면 사이의 틈을 재는 값이다. 두 값은 다음과 같이 다르다."));
body.push(Gap(60));
body.push(Tbl(
  ["조건", "eq4 전역 평면 잔차", "eq6 3m 직선자 처짐"],
  [
    ["전역 휨 12mm (완만)", "7.90 mm", "0.54 mm"],
    ["국소 융기 6mm (급함)", "0.72 mm", "4.67 mm"],
  ],
  [3400, 3100, 3138]));
body.push(P("따라서 eq4 잔차로는 합격·불합격을 말할 수 없다. eq4는 요철의 위치를 찾는 데 쓰고, 시방 판정은 eq6이 담당한다.", { before: 140 }));
body.push(P("구간 안에서 자가 닿는 자리는 프로파일의 상부 볼록껍질이며, 처짐량은 그 껍질과 표면 사이 최대 간격이다. 이 정의는 볼록한 돌출과 오목한 함몰을 모두 물리적으로 맞게 다룬다."));
body.push(Note("분해능 편향 진단: 요철 폭이 평활 반경보다 좁으면 처짐량이 반드시 낮게 나온다. 이는 점 밀도의 물리적 한계이나, 기준초과를 합격으로 내보내는 방향이라 위험하다(실측: GT 8mm 융기 → 처짐 5.6mm → 허용 7mm 대비 거짓 합격). 평활을 촘촘히/성기게 두 번 걸어 결과 차이를 편향 추정치로 쓰고, 편향을 더해 허용치를 넘으면 「판정보류(분해능)」로 내보내 거짓 합격을 막는다."));

/* ================= 4. 입력 데이터 ================= */
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("4. 입력 데이터 — 매 촬영 vs 캘리브레이션"));
body.push(P("검측 알고리즘이 동작하려면 하드웨어가 정해진 데이터를 정확히 제공해야 한다. 데이터는 성격이 전혀 다른 두 갈래로 나뉜다. 매 촬영마다 새로 들어오는 것(A)과, 출고 시 1회 측정해 모듈에 저장한 뒤 계속 쓰는 것(B)이다. PDF 3.1이 「B의 정확도가 곧 측정 정확도」라고 못 박은 대로, A가 아무리 정확해도 B가 틀리면 결과가 통째로 틀어진다."));
body.push(P("아래 표의 「출처」 열은 값의 근거를 세 등급으로 구분한 것이다. 이 구분이 현재 시스템의 신뢰도 상한을 그대로 보여준다.", { before: 100 }));
body.push(Bul("spec — PDF 하드웨어 사양에서 유도. 부품 번호가 있는 실물 기준"));
body.push(Bul("design — 설계 고정값. 용역서에 명시되어 있으나 조립 후 실측 필요"));
body.push(Bul("assumed — 가정값. 출고 전 실측으로 대체해야 하는 항목"));

body.push(H2("4.1 매 촬영 데이터 (A)"));
body.push(P("촬영할 때마다 하드웨어가 새로 보내는 데이터다. 세 가지뿐이다."));
body.push(Gap(60));
body.push(Tbl(
  ["데이터", "기호", "용도", "현재 상태", "비고"],
  [
    ["레이저 ON 프레임", M("rgb_on"), "격자 선검출", "런타임", "무손실 포맷(PNG/RAW) 권장"],
    ["레이저 OFF 프레임", M("rgb_off"), "영역 분할·차영상", "런타임", "ON과 수십 µs 간격"],
    ["격자점 픽셀 좌표", M("u, v"), "삼각측량 입력", "런타임", "A_선검출 출력"],
    ["IMU 중력벡터", M("ĝ"), "수직·수평도 기준", { t: "미주입", c: C.warn },
     { t: "자세별 기본값으로 대체 중." + SEP + "실장비는 촬영 순간 가속도계 값 필요", c: C.warn }],
  ],
  [2050, 1000, 1900, 1500, 3188]));

body.push(H2("4.2 캘리브레이션 데이터 (B)"));
body.push(P("출고 시 1회 측정해 모듈에 저장하는 값이다. 촬영마다 바뀌지 않지만, 틀리면 모든 촬영이 함께 틀린다."));
body.push(Gap(60));
body.push(Tbl(
  ["데이터", "기호", "현재 값", "출처", "비고"],
  [
    ["카메라 초점거리", M("f"), M("2318.8 px"), { t: "design", c: C.warn },
     { t: "PDF 에 초점거리 값이 없다. 8mm 는 격자" + SEP + "수용·고정초점 심도·목표 정밀도에서 유도", c: C.warn }],
    ["주점", M("c_x, c_y"), M("1224.0, 1024.0"), { t: "assumed", c: C.warn },
     { t: "센서 정중앙 가정. 실제 주점은" + SEP + "제조 공차로 수십 px 어긋난다", c: C.warn }],
    ["기선", M("b"), M("0.150 m"), "spec",
     "PDF 2.2 카메라–레이저 광축 150mm." + SEP + "조립 후 실측 필요"],
    ["DOE 발산각", "—", M("42.61°"), "spec",
     "PDF 2.2 「120cm 에서 936×936mm」" + SEP + "= 2·atan(468/1200)"],
    ["격자선 수", "—", M("수직20 + 수평20"), "spec",
     "PDF 2.2 400 교점 정방형 격자"],
    ["레이저 수렴각", M("δ"), M("6.18°"), "design",
     "격자를 센서 중앙에 오게 하는 설계값." + SEP + "시차 이동량을 상쇄한다"],
    ["V선 발사각", M("α_i"), M("사인등간격 20분할"), { t: "assumed", c: C.warn },
     { t: "회절격자 sin θ_m = m·λ/d 를 반영한" + SEP + "모델값. 실측 α_i 로 대체 필요", c: C.warn }],
    ["H선 발사각", M("β_j"), M("사인등간격 20분할"), { t: "assumed", c: C.warn },
     { t: "동일", c: C.warn }],
    ["카메라–레이저 자세", M("R, t"), M("R=I, t=(b,0,0)"), { t: "assumed", c: C.warn },
     { t: "두 광축이 완전 평행하고 좌우" + SEP + "오프셋만 있다고 가정", c: C.warn }],
    ["IMU–카메라 자세", M("R_ic"), M("단위행렬"), { t: "assumed", c: C.warn },
     { t: "IMU가 카메라와 정렬됐다고 가정", c: C.warn }],
    ["가속도계 bias", M("b_a"), M("미구현"), { t: "assumed", c: C.warn },
     { t: "중력벡터 정밀도 보정 미적용", c: C.warn }],
    ["선검출 픽셀오차", M("σ_u"), M("0.2 px"), { t: "assumed", c: C.warn },
     { t: "불확실도 산정의 기준. 실장비" + SEP + "반복성 측정 필요", c: C.warn }],
  ],
  [1900, 950, 1500, 1000, 4288]));
body.push(Note("PDF 2.2 사양표에서 그대로 오는 값은 기선 150mm, DOE 발산각 42.61°, 격자선 20+20 세 가지다. 초점거리는 PDF 가 「저왜곡 F2.0급, 초점 ~1.2m 고정」이라고만 쓰고 값을 주지 않아 이 코드가 8mm 로 정했다(근거는 4.4). 나머지 일곱 항목은 가정값이며, 이것이 남아 있는 한 절대 정확도를 보증할 수 없다. 특히 α_i 는 모델을 잘못 고르면 1.2m 에서 깊이가 32mm 어긋나므로 출고 전 실측이 선택이 아니라 필수다."));

body.push(H2("4.3 함수별 필수 입력 (코드 기준)"));
body.push(Tbl(
  ["식", "함수", "필수 입력", "주요 기본값"],
  [
    ["①", M("eq1.triangulate_point"), M("u, v, alpha, beta," + SEP + "f, b, cx, cy"), "없음 (전부 필수)"],
    ["②", M("eq2.fit_plane_tls_ransac"), M("points_3d"), M("threshold=0.005 m" + SEP + "max_trials=300")],
    ["②", M("eq2.fit_axis_ransac"), M("points_3d"), M("radius_m=0.06" + SEP + "min_slenderness=13")],
    ["③", M("eq3.measure_from_gravity"), M("vec, g_hat, kind"), "없음"],
    ["③", M("eq3.gravity_in_laser_frame"), M("g_imu"), M("R_ic=None, R_cl=None" + SEP + "(둘 다 단위행렬)")],
    ["④", M("eq4.detect_defects_region"), M("points_3d"), M("threshold_mm=1.5" + SEP + "grid_n=24, window='auto'")],
    ["⑤", M("eq5.region_uncertainty"), M("points_3d," + SEP + "camera_params"), M("sigma_u_px=0.2" + SEP + "target_sigma_mm=2.0")],
    ["⑥", M("eq6.straightedge_gap"), M("points_3d"), M("length_m=3.0" + SEP + "n_directions=4")],
  ],
  [700, 3000, 3000, 2938]));

/* ================= 5. 파라미터 ================= */
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("5. 현재 파라미터 설정값"));

body.push(H2("5.1 캘리브레이션 상수"));
body.push(P("calibration.py 가 단일 출처다. 이전에는 inspection.py 와 synth_scene.py 가 같은 값을 따로 들고 있어, 한쪽만 고치면 조용히 어긋났다.", { after: 80 }));
body.push(Note("사양 프로파일이 셋이다. legacy(원본 v4, 현재 기본값) · pdf(PDF 사양) · improved(정확도 개선안). 실제 하드웨어 사양이 확정되지 않았고 이미 뽑아 둔 Isaac 렌더가 legacy 값으로 만들어졌으므로 기본값을 legacy 로 둔다. 5.1~5.3 은 pdf 프로파일, 5.4 는 improved 기준이다. calibration.SPEC_PROFILES 에 셋 다 들어 있고 환경변수 LASER_GRID_PROFILE 로 전환한다."));
body.push(Gap(60));
body.push(Tbl(
  ["", "legacy (기본)", "pdf", "improved"],
  [
    ["출처", { t: "원본 v4 튜닝값", c: C.warn }, "PDF 2.2 사양표", "PDF + 정확도 유도"],
    ["f_px", "1593.0", "2318.8", "2919.7"],
    ["DOE 발산각", "60.82 °", "42.61 °", "42.61 °"],
    ["격자선", "V21 + H21", "V20 + H20", "V40 + H20"],
    ["기선", "150 mm", "150 mm", "180 mm"],
    ["센서", "컬러 2448×2048", "컬러 2448×2048", "모노 3072×2560 + 520nm"],
    ["레이저 수렴각", "0 °", "6.18 °", "7.40 °"],
    ["발사각 분포", "각도 등간격", "사인 등간격", "사인 등간격"],
    ["격자 피치 @1.2m", "70.4 mm", "49.3 mm", { t: "24.0 mm", c: C.accent }],
    ["σ_Z @1.2m", "1.205 mm", "0.828 mm", { t: "0.274 mm", c: C.accent }],
    ["센서 가장자리 여유", { t: "50 px", c: C.warn }, "120 px", "141 px"],
  ],
  [2000, 2200, 2200, 2638]));
body.push(Note("검측식(eq1~eq6)은 이 값들에 의존하지 않는다. 삼각측량식이 f·b·α 를 인자로 받을 뿐이므로, 실제 하드웨어 사양이 들어오면 SPEC_PROFILES 딕셔너리 한 줄만 바꾸면 된다. 회귀 검증은 세 프로파일 모두에서 전체 통과한다."));
body.push(...Code([
  "PIXEL_PITCH_UM = 3.45      # spec   PDF 「픽셀 >=3.45um」",
  "IMAGE_W/H      = 2448/2048 # spec   PDF 5MP 글로벌셔터",
  "BASELINE_M     = 0.150     # spec   PDF 광축 150mm",
  "LENS_FNUMBER   = 2.0       # spec   PDF 「저왜곡 F2.0급」",
  "FOCUS_DIST_M   = 1.2       # spec   PDF 「초점 ~1.2m 고정 잠금」",
  "N_VERTICAL/H   = 20 / 20   # spec   PDF 400 교점",
  "FOV_DEG        = 42.61     # spec   PDF 120cm 936mm 에서 유도",
  "WORK_Z_MIN/MAX = 1.0 / 1.5 # spec   PDF 권장 측정거리",
  "",
  "LENS_FOCAL_MM  = 8.0       # design PDF 에 값 없음 (5.3 참조)",
  "LASER_TILT_DEG = 6.18      # design 격자를 센서 중앙에 두는 수렴각",
  "EDGE_MARGIN_PX = 50        # design 센서 가장자리 여유",
  "",
  "f_px  = 8.0mm / 3.45um    = 2318.8 px       (유도)",
  "cx,cy = 2448/2, 2048/2    = 1224.0, 1024.0  (가정 - 센서 중앙)",
]));
body.push(P("카메라 HFOV 55.65°, VFOV 47.65°. 거리별 시야·격자 투사폭·깊이 노이즈는 다음과 같다.", { before: 120 }));
body.push(Gap(60));
body.push(Tbl(
  ["거리", "카메라 시야", "격자 투사폭", "분해능", "σ_Z (σ_u=0.2px)"],
  [
    ["0.5 m", "528 × 442 mm", "390 mm", "0.216 mm/px", { t: "0.14 mm", c: C.accent }],
    ["1.0 m", "1056 × 883 mm", "780 mm", "0.431 mm/px", { t: "0.58 mm", c: C.accent }],
    ["1.2 m", "1267 × 1060 mm", "936 mm", "0.518 mm/px", { t: "0.83 mm", c: C.accent }],
    ["1.5 m", "1584 × 1325 mm", "1170 mm", "0.647 mm/px", { t: "1.29 mm", c: C.accent }],
    ["2.0 m", "2111 × 1766 mm", "1560 mm", "0.863 mm/px", { t: "2.30 mm", c: C.warn }],
  ],
  [900, 2400, 1600, 1700, 3038]));
body.push(Note("PDF 5.2 는 「기선/거리 조합만으로는 σ_Z ≤ 2mm 목표 달성 불가」라고 결론냈다. 그것은 시뮬레이션 튜닝값 f=1593px 기준이며, 실제 부품 사양에서 나오는 f=2318.8px 로는 권장 측정거리 1.5m 까지 1.29mm 로 목표를 만족한다. 2.0m 를 넘어가면 다시 목표를 벗어나므로, PDF 가 측정거리를 1.5m 로 제한한 것은 타당하다."));

body.push(H2("5.2 격자 — 시야 안에 담기는 설계"));
body.push(P("격자의 이미지상 위치는 발산각만으로 정해지지 않는다. 카메라가 레이저에서 기선 b 만큼 떨어져 있으므로, 격자 전체가 거리에 따라 좌우로 이동한다."));
body.push(...Code([
  "u = f·tan(α) − f·b/Z + c_x",
  "",
  "  시차 이동량 f·b/Z",
  "    Z=0.5m →  696 px  (센서 폭의 28.4%)",
  "    Z=1.0m →  348 px  (14.2%)",
  "    Z=1.2m →  290 px  (11.8%)",
  "    Z=1.5m →  232 px  ( 9.5%)",
]));
body.push(P("이 이동량 때문에 레이저 축을 카메라와 평행하게 두면 센서 한쪽이 통째로 낭비된다. 레이저를 카메라 쪽으로 조금 기울이면 격자가 작업거리에서 화면 중앙에 오므로 양쪽을 고르게 쓸 수 있다. 수렴각 6.18° 는 작업거리 양 끝(1.0m·1.5m)에서 좌우 네 여유 중 최소값이 최대가 되는 값을 0.01° 간격으로 찾은 것이다.", { before: 120 }));
body.push(Gap(60));
body.push(Tbl(
  ["레이저 축 배치", "1.0m 좌측 여유", "1.5m 우측 여유", "판정"],
  [
    ["카메라와 평행 (δ = 0)", { t: "−98 px", c: C.warn }, { t: "656 px", c: C.ink3 },
     { t: "근거리에서 격자 왼쪽이 잘림", c: C.warn }],
    ["6.18° 수렴 (채택)", { t: "249 px", c: C.accent }, { t: "250 px", c: C.accent },
     { t: "양 끝 모두 마진 50px 확보", c: C.accent }],
  ],
  [2400, 1800, 1800, 3638]));
body.push(P("삼각측량 기하는 그대로이고 발사각의 기준축만 바뀌므로, α_i 에 수렴각을 더해 쓰면 된다(탄젠트 덧셈정리로 정확히 성립). β 는 1/(cosδ − sinδ·tanα) 만큼 살짝 결합되어 격자가 미세한 사다리꼴이 되지만, 깊이 Z 는 α 와 u 로만 정해지므로 영향이 없고 H선 예측 위치에만 반영된다.", { before: 140 }));

body.push(H3("정합성 검사 결과"));
body.push(...Code([
  "작업거리 1.0 ~ 1.5 m,  마진 50px",
  "  Z=1.0m   u =  249.4 .. 2082.5    (허용 50 .. 2398)",
  "  Z=1.5m   u =  365.3 .. 2198.5",
  "  v      =  119.7 .. 1928.3        (허용 50 .. 1998)",
  "",
  "  판정: 격자 전부 센서 안",
  "  이 설계로 쓸 수 있는 거리: 0.64 ~ 10.75 m",
  "  고정초점 1.2m 피사계심도: 0.95 ~ 1.61 m  (작업거리 포함)",
]));

body.push(H2("5.3 렌즈 초점거리 — PDF 가 값을 주지 않은 유일한 광학 상수"));
body.push(P("PDF 2.2 는 렌즈를 「저왜곡 F2.0급, 초점 ~1.2m 고정 잠금」이라고만 쓰고 초점거리를 명시하지 않는다. 이전 버전은 12mm 를 「PDF 사양」이라고 적어 두었으나 그런 값은 PDF 에 없다. 나머지 사양이 초점거리를 사실상 하나로 몰아준다."));
body.push(Gap(60));
body.push(Tbl(
  ["초점거리", "f_px", "격자 42.61° 수용", "고정초점 1.2m 심도", "σ_Z @1.5m", "판정"],
  [
    ["6 mm", "1739", "들어옴", "0.82 ~ 2.21 m", "1.73 mm", { t: "가능", c: C.ink3 }],
    ["8 mm", { t: "2319", c: C.accent }, { t: "들어옴", c: C.accent },
     { t: "0.95 ~ 1.61 m", c: C.accent }, { t: "1.29 mm", c: C.accent },
     { t: "채택", c: C.accent }],
    ["10 mm", "2899", { t: "벗어남", c: C.warn }, { t: "1.03 ~ 1.44 m 부족", c: C.warn }, "1.04 mm", { t: "탈락", c: C.warn }],
    ["12 mm", "3478", { t: "벗어남", c: C.warn }, { t: "1.08 ~ 1.35 m 부족", c: C.warn }, "0.86 mm", { t: "탈락", c: C.warn }],
    ["16 mm", "4638", { t: "벗어남", c: C.warn }, { t: "1.13 ~ 1.28 m 부족", c: C.warn }, "0.65 mm", { t: "탈락", c: C.warn }],
  ],
  [1100, 900, 1700, 2100, 1300, 2538]));
body.push(P("세 조건이 각각 다른 방향에서 초점거리를 조인다.", { before: 140, after: 60 }));
body.push(...Code([
  "(1) 격자 수용 — DOE 42.61° 는 고정이므로 세로 시야가 먼저 막힌다",
  "      f <= (H/2 - margin)·pitch / tan(21.31°) = 8.62mm",
  "",
  "(2) 고정초점 심도 — 초점 조절 장치가 없으므로 1.0~1.5m 를 덮어야 한다",
  "      착란원 2px(6.9um) 기준,  8mm F2.0 → 0.95 ~ 1.61 m   덮음",
  "                              12mm F2.0 → 1.08 ~ 1.35 m   양 끝 벗어남",
  "",
  "(3) 목표 정밀도 — sigma_Z = sigma_u·Z^2/(f·b) <= 2mm @1.5m",
  "      8mm 에서 1.29mm.  6mm 에서도 1.73mm 로 만족",
]));
body.push(Note("6mm 도 세 조건을 통과하지만 8mm 쪽이 깊이 노이즈가 25% 작고 격자가 화면을 더 크게 채운다. 8mm 를 넘어가면 (1)과 (2)가 동시에 깨진다. 즉 PDF 사양을 모두 지키는 표준 초점거리는 사실상 8mm 하나다. 2/3″ 커버·저왜곡·F2.0급 8mm 렌즈는 머신비전에서 흔한 규격이므로 조달에 문제가 없다."));
body.push(Note("PDF 2.2 카메라 항의 「1.2m·FOV 200mm 에서 ≈0.08mm/px」는 이 설계와 맞지 않는다. 1.2m 에서 시야 200mm 를 만들려면 초점거리가 약 50mm 여야 하는데, 그러면 화각이 9.5° 로 좁아져 격자 936mm 를 담을 수 없고 심도도 성립하지 않는다. 이 줄은 근접 확대 촬영을 상정한 별도 수치로 보이며 본 설계에는 반영하지 않았다. 확인이 필요하다."));

body.push(H3("발사각"));
body.push(P("발산각 42.61° 를 20분할한 뒤 V선에는 수렴각 6.18° 를 더한 값이다. 분할은 각도 등간격이 아니라 사인 등간격이다. 회절격자의 m 차 광은 sin θ_m = m·λ/d 를 따르기 때문이다."));
body.push(...Code([
  "α0 =-15.126°  α1 =-12.792°  α2 =-10.490°  α3 = -8.215°",
  "α4 = -5.963°  α5 = -3.731°  α6 = -1.513°  α7 =  0.693°",
  "α8 =  2.891°  α9 =  5.084°  α10=  7.276°  α11=  9.469°",
  "α12= 11.667°  α13= 13.873°  α14= 16.091°  α15= 18.323°",
  "α16= 20.575°  α17= 22.850°  α18= 25.152°  α19= 27.486°",
  "",
  "간격 2.192° (중앙) ~ 2.334° (가장자리).  등각도가 아니다",
  "β_j 는 수렴각 없이 -21.306° .. +21.306°",
]));
body.push(Note("등각도로 두면 같은 발산각·같은 포락선에서도 안쪽 선의 위치가 최대 0.19° 어긋난다. 각도로는 작아 보이지만 깊이로 환산하면 1.2m 에서 32mm 다(dZ/dα ≈ Z²/b). 목표 ±2mm 의 16배이므로 모델 선택만으로는 부족하고, 출고 시 DOE 실측 α_i 를 받아 쓰는 것이 필수다."));

body.push(Note("발사각이 여전히 가장 취약한 값이다. PDF 3.1은 「i=0~20 각각 다른 α_i 가 DOE 제조 시 이미 고정」이라고 명시하는데, 코드는 균등 분할로 생성한다. DOE 회절은 sinθ_m = m·λ/d 를 따라 등각도가 아니라 등 sin(각)에 가까우므로 외곽선일수록 벌어진다. 출고 전 평면 타깃 실측으로 대체해야 한다."));

body.push(H2("5.4 정확도 개선안"));
body.push(P("PDF 사양은 목표 정확도를 만족한다. 다만 여유가 항목마다 크게 다르다. 합성 씬 8회로 측정한 현행 오차는 다음과 같다."));
body.push(Gap(60));
body.push(Tbl(
  ["검측 항목", "목표", "PDF 원안 오차", "목표 대비 여유"],
  [
    ["면 수직·수평도", "±0.5°", "0.013 °", { t: "1/38 — 넉넉", c: C.accent }],
    ["동바리 수직도", "±0.5°", "0.149 °", { t: "1/3 — 빠듯", c: C.warn }],
    ["평활도 불확실도", "±2 mm", "±0.59 mm", { t: "1/3 — 빠듯", c: C.warn }],
    ["깊이 잡음 σ_Z @1.2m", "—", "0.83 mm", "—"],
  ],
  [2400, 1400, 1800, 2438]));
body.push(P("뒤의 두 항목이 병목이고 원인이 같다. 1.2m 에서 격자 피치가 49.3mm 인데 동바리 지름이 Ø48.6mm 라, 부재 하나에 V선이 한두 개밖에 걸리지 않는다. 벽 프로파일도 49mm 간격으로만 찍힌다. 즉 정확도를 올리는 첫 지렛대는 광학 정밀도가 아니라 공간 표본 밀도다.", { before: 140 }));

body.push(H3("깊이 잡음이 무엇으로 정해지는가"));
body.push(P("초점거리를 지렛대로 보기 쉬우나 그렇지 않다. 화각을 고정하면 f_px = N_화소 · Z / W_시야 이므로 초점거리가 소거된다."));
body.push(...Code([
  "σ_Z = σ_u · Z² / (f_px · b)",
  "    = σ_u · Z · W_시야 / (N_화소 · b)",
  "",
  "  담을 면적 W 를 정하면 초점거리는 따라온다.",
  "  남는 지렛대는  σ_u · N_화소 · b  세 개뿐이다.",
]));

body.push(H3("바꾸는 것"));
body.push(Gap(60));
body.push(Tbl(
  ["항목", "PDF 원안", "개선안", "근거"],
  [
    ["격자선", "V20 + H20", { t: "V40 + H20", c: C.accent },
     "깊이를 주는 것은 V선뿐이다. 기선이 X축이라" + SEP +
     "H선은 시차가 선과 나란해 삼각측량이 풀리지" + SEP +
     "않는다. H선을 늘려도 측정점은 늘지 않는다"],
    ["센서", "컬러", { t: "모노", c: C.accent },
     "베이어 배열은 녹색선이 적·청 화소에서 신호를" + SEP +
     "잃고 디모자이크가 선 단면을 뭉갠다"],
    ["광학필터", "없음", { t: "520nm 대역통과", c: C.accent },
     "배경광 약 1/30. 직사광 아래 SNR 확보." + SEP +
     "모노와 합쳐 σ_u 0.2 → 0.1px"],
    ["화소", "3.45 µm / 2448", { t: "2.74 µm / 3072", c: C.accent },
     "같은 2/3″ 에 작은 화소. 렌즈·화각·심도는" + SEP +
     "그대로 두고 화소 수만 1.26배"],
    ["기선", "150 mm", { t: "180 mm", c: C.accent },
     "σ_Z ∝ 1/b. 외형 210mm 폭 안에서 렌즈" + SEP + "경통 여유를 남긴 최대값"],
    ["레이저 수렴각", "6.18 °", { t: "7.40 °", c: C.accent },
     "화소 수가 바뀌어 최적값 재탐색"],
  ],
  [1400, 1500, 1600, 5138]));

body.push(H3("바꾸지 않는 것"));
body.push(Bul("DOE 발산각 42.61° — 좁히면 σ_Z 는 좋아지지만 담는 면적이 준다. KCS 3m 직선자 기준은 이미 한 장으로 커버가 안 되므로(1.5m 에서 1.17m) 화각을 줄이는 방향은 손해다."));
body.push(Bul("렌즈 8mm F2.0 초점 1.2m 고정 — 심도가 작업거리를 덮는 유일한 값이다(5.3)."));
body.push(Bul("레이저 출력 30~49mW — 선이 40 → 60개로 늘어 선당 광량은 0.67배가 되지만, 모노(약 2배)와 대역통과 필터(배경 1/30)가 그 이상을 보상한다. 출력을 올리지 않으므로 눈 안전등급 재평가가 필요 없다."));
body.push(Bul("측정거리 1.0~1.5m — PDF 권장 유지. 심도가 0.997~1.507m 로 이 구간을 덮는다(여유가 얇으므로 F2.4 로 조이면 넓어진다)."));

body.push(H3("측정된 개선 효과"));
body.push(P("experiment_spec.py 가 같은 합성 씬을 두 프로파일로 8회씩 통과시킨 결과다.", { after: 60 }));
body.push(Tbl(
  ["항목", "단위", "PDF 원안", "개선안", "개선"],
  [
    ["깊이 잡음 σ_Z @1.2m", "mm", "0.828", { t: "0.274", c: C.accent }, "3.02배"],
    ["격자 피치 @1.2m", "mm", "49.26", { t: "24.00", c: C.accent }, "2.05배"],
    ["벽 수직도 오차", "°", "0.0126", "0.0129", { t: "변화 없음", c: C.ink3 }],
    ["바닥 수평도 오차", "°", "0.0165", { t: "0.0028", c: C.accent }, "5.83배"],
    ["동바리 수직도 (평균)", "°", "0.1494", { t: "0.0284", c: C.accent }, "5.27배"],
    ["동바리 수직도 (최악)", "°", "0.1563", { t: "0.0553", c: C.accent }, "2.83배"],
    ["직선자 처짐 오차", "mm", "0.543", { t: "0.082", c: C.accent }, "6.62배"],
    ["직선자 불확실도 밴드", "mm", "0.586", { t: "0.125", c: C.accent }, "4.69배"],
    ["마스크 +16px 훼손 시 최악", "°", "0.2256", { t: "0.0353", c: C.accent }, "6.40배"],
  ],
  [2500, 800, 1400, 1400, 2938]));
body.push(Note("벽 수직도만 개선되지 않는다. 두 사양 모두 0.013° 로 목표 ±0.5° 의 1/38 이며, 수천 점을 평균한 결과라 이미 수치 바닥이다. 사양을 올려도 여기서는 얻을 것이 없다. 실제로 좁던 동바리 각도와 평활도는 원인이 같았고(격자 피치), 그 하나를 고치자 세그멘테이션 훼손 강건성까지 함께 올라갔다 — 표본이 많을수록 경계 오염이 평균에 묻힌다."));
body.push(Note("사양 프로파일은 calibration.SPEC_PROFILES 에 있고, 환경변수 LASER_GRID_PROFILE=pdf 로 원안으로 되돌릴 수 있다. 회귀 검증은 두 프로파일 모두에서 전체 통과한다."));

body.push(H3("대가와 확인이 필요한 것"));
body.push(Gap(60));
body.push(Tbl(
  ["항목", "내용"],
  [
    ["RGB 문맥 영상 상실", "모노 센서 + 대역통과 필터는 색을 남기지 않는다. PDF 3.1 의 「RGB 영상, 컬러」 요구와 어긋나며, 세그멘테이션을 흑백으로 해야 한다. 기하 전용(geom) 백엔드는 색을 쓰지 않으므로 영향이 없고, VLM/SAM 백엔드는 흑백 입력으로 성능 확인이 필요하다."],
    ["OFF 프레임 노출", "대역통과 필터는 배경광을 1/30 으로 줄이므로, 문맥 영상용 OFF 프레임은 노출을 그만큼 늘려야 한다. 글로벌 셔터에서 10ms 수준이면 핸드헬드로도 무리가 없으나 실측 확인이 필요하다."],
    ["「픽셀 >=3.45µm」 조건 위배", "PDF 2.2 는 화소를 3.45µm 이상으로 못박았다. 2.74µm 는 이 조건에서 벗어난다. 화소가 작아지면 화소당 광량이 줄지만, 모노(약 2배)와 대역통과 필터가 이를 덮는다는 것이 전제다. 실장비 노출 시험으로 확인해야 한다."],
    ["기선 180mm 강성", "PDF 는 「강성 고정, 측정 중 변형 없을 것」을 요구한다. 기선이 길어질수록 같은 변형각이 더 큰 오차가 된다. 외형 210mm 안에서 30mm 를 늘리는 만큼 마운트 강성 검토가 필요하다."],
    ["수렴각 7.40° 기구 반영", "레이저 축을 카메라 쪽으로 7.40° 기울여야 한다. 조립 후 실제 각도를 캘리브레이션으로 다시 측정해야 한다."],
  ],
  [2000, 7038]));

body.push(H2("5.5 판정 기준"));
body.push(H3("수직·수평도 — eq3.KCS_SPEC"));
body.push(Tbl(
  ["부재 분류", "tol_mm", "tol_ratio", "tol_deg"],
  [
    ["wall / column / shoring / rebar", "20.0", "1/1000 (권장 병기)", "0.5"],
    ["formwork / floor / slab", "20.0", "—", "0.5"],
    ["masonry", "10.0", "—", "0.5"],
  ],
  [4200, 1600, 2200, 1638]));
body.push(P("h/1000은 PDF 표에서 「층고대비 권장」으로 병기된 값이므로 본판정을 덮어쓰지 않고 권장기준으로만 함께 보고한다.", { before: 100, size: 18, color: C.ink2 }));

body.push(H3("평활도 직선자 — eq6.KCS_FLATNESS_SPEC"));
body.push(Tbl(
  ["부재 분류", "직선자 길이 · 허용 처짐량", "근거"],
  [
    ["wall / formwork_wall / formwork_column", "3.0 m · 7.0 mm", "KCS 14 20 10"],
    ["plaster_wall / masonry", "1.0 m · 10.0 mm", "KCS 41 46 00"],
    ["floor / slab", "1.0 m · 10.0 mm 및 3.0 m · 10.0 mm", "KCS 14 20 10"],
    ["ceiling", "3.0 m · 3.0 mm", "KCS 41 52 00"],
  ],
  [3600, 3800, 2238]));

body.push(H2("5.6 알고리즘 튜닝값"));
body.push(...Code([
  "eq5  erode_default_px      3       마스크 침식 (면)",
  "     erode_thin_px         1       마스크 침식 (동바리·철근)",
  "     min_points           12       영역 최소 점 수",
  "     jump_ratio         0.05       깊이 불연속 상대 임계",
  "     min_jump_m         0.02 m     깊이 불연속 절대 하한",
  "     linear_ratio       0.15       λ2/λ1 — 선형 판정",
  "     planar_ratio       0.15       λ3/λ2 — 평면 판정",
  "     thin_extent_m      0.12 m     선형 부재 최대 횡폭",
  "     align_deg          30.0       중력 정렬 허용각",
  "     min_thickness_ratio 0.02      λ3/λ2 — 판 vs 원통 판별",
  "",
  "eq2  plane threshold     0.005 m   TLS RANSAC inlier",
  "     axis radius_m       0.06 m    축 주변 inlier 반경",
  "     min_inlier_frac     0.45      축 적합 유효 하한",
  "",
  "eq4  threshold_mm        1.5       요철 판정 임계",
  "     grid_n              24        면내 격자 분할 수",
  "     window            'auto'      평활 창 자동 결정",
  "",
  "eq6  n_directions        4         자를 얹는 방향 수",
  "     target_bin_points   8         평활 창 안 목표 점 수",
  "     fine/coarse       4 / 16      분해능 편향 진단용 두 척도",
  "",
  "pipe split outlier_frac  0.08      병합 영역 재분할 발동",
  "     plane_threshold_m   0.015 m",
  "     sigma_u_px          0.2 px    불확실도 산정",
  "     target_sigma_mm     2.0 mm    평활도 목표 정밀도",
]));

body.push(H2("5.7 값 변경 내역"));
body.push(Tbl(
  ["항목", "이전", "현재", "근거"],
  [
    ["렌즈 초점거리", { t: "12 mm (사양 아님)", c: C.warn }, { t: "8 mm", c: C.accent },
     "PDF 는 초점거리를 명시하지 않는다." + SEP + "12mm 로는 격자도 심도도 안 맞는다 (5.3)"],
    ["초점거리 f_px", { t: "3478.3", c: C.ink3 }, { t: "2318.8", c: C.accent },
     "8mm ÷ 화소 3.45µm." + SEP + "그 이전 1593 은 5.50mm 에 해당했다"],
    ["DOE 발산각", { t: "31.0° (임의)", c: C.warn }, { t: "42.61°", c: C.accent },
     "PDF 2.2 「120cm 936×936mm」." + SEP + "8mm 렌즈로 그대로 담긴다"],
    ["격자 선 수", { t: "21 + 21", c: C.ink3 }, { t: "20 + 20", c: C.accent },
     "PDF 2.2 「수직20 + 수평20," + SEP + "400 교점」"],
    ["발사각 분포", { t: "각도 등간격", c: C.ink3 }, { t: "사인 등간격", c: C.accent },
     "회절격자 sin θ_m = m·λ/d." + SEP + "차이가 1.2m 깊이 32mm 에 해당"],
    ["레이저 수렴각", { t: "5.1°", c: C.ink3 }, { t: "6.18°", c: C.accent },
     "작업거리 양 끝 여유가 최대가" + SEP + "되는 값 (탐색)"],
    ["Isaac 카메라 설정", { t: "aperture 36mm" + SEP + "focal 51.15mm", c: C.ink3 },
     { t: "aperture 8.4456mm" + SEP + "focal 8mm", c: C.accent },
     "센서 실물 크기를 넣어야 f-stop·" + SEP + "심도까지 사양과 같아진다"],
    ["기선 b", "0.150 m", "0.150 m", "유지 (PDF 2.2 사양값)"],
    ["주점 c_x, c_y", "1224, 1024", "1536, 1280", "센서 중앙 가정 (해상도 변경 반영)"],
    ["사양 프로파일", { t: "없음", c: C.ink3 }, { t: "pdf / improved", c: C.accent },
     "PDF 원안과 개선안을 둘 다 코드에" + SEP + "남기고 환경변수로 전환 (5.4)"],
    ["가림 그림자 병합", { t: "없음", c: C.ink3 }, { t: "25cm 이내 조각 병합", c: C.accent },
     "격자를 조밀하게 하자 동바리 그림자가" + SEP + "벽을 둘로 쪼갰다 (아래)"],
  ],
  [1700, 1500, 1500, 4938]));

body.push(H3("가림 그림자로 갈라진 벽 병합"));
body.push(P("격자를 조밀하게 만들자 예상하지 못한 곳이 깨졌다. 영역 분할의 DBSCAN eps 는 점 밀도에 맞춰 자동으로 좁아지는데, V선을 20 → 40 개로 늘리자 eps 도 함께 좁아져 앞에 선 동바리가 벽에 드리운 폭 5cm 짜리 빈 띠가 「서로 다른 두 벽」으로 보이게 되었다."));
body.push(Gap(60));
body.push(Tbl(
  ["", "벽 영역 수", "직선자 프로파일 길이", "직선자 처짐"],
  [
    ["병합 전 (geom)", { t: "2 개", c: C.warn }, { t: "0.69 m", c: C.warn },
     { t: "2.58 mm", c: C.warn }],
    ["병합 후 (geom)", { t: "1 개", c: C.accent }, { t: "1.15 m", c: C.accent },
     { t: "3.91 mm", c: C.accent }],
    ["참값", "1 개", "1.15 m", "3.99 mm"],
  ],
  [2200, 1600, 2400, 2000]));
body.push(P("각도는 두 조각 모두 같은 법선을 주므로 멀쩡했고, 그래서 평활도만 조용히 틀렸다. 가림 그림자는 좁고(부재 지름 정도) 개구부·벽 분리는 넓다는 점을 이용해, 같은 평면에서 나온 조각 중 최근접 거리가 25cm 미만인 것만 다시 합친다. Ø48.6mm 동바리의 그림자는 10cm 안쪽, 문 개구부는 80cm 이상이라 둘이 섞이지 않는다.", { before: 120 }));
body.push(Note("이번 개정의 핵심은 「PDF 에 있는 값」과 「없어서 정한 값」을 분리한 것이다. 이전 표는 렌즈 12mm 를 spec 으로 적어 두었으나 PDF 에 그런 줄이 없고, 12mm 를 전제로 DOE 발산각을 42.61° 에서 31.0° 로 임의로 낮춰 놓은 상태였다. PDF 에 실제로 적힌 것은 DOE 쪽(936mm, 20+20, 400교점)이므로, 그것을 고정하고 명시되지 않은 초점거리를 8mm 로 정하는 것이 사양을 지키는 방향이다."));

body.push(H3("기하 예측의 시차항 누락 수정"));
body.push(P("A_선검출의 격자 위치 예측이 u = f·tan(α) + c_x 로, 시차항 −f·b/Z 를 빼먹고 있었다. 예측은 추적 밴드의 중심을 잡는 데 쓰이는데 밴드 폭이 20~50px 이므로, 수백 px 어긋난 예측은 밴드를 실제 선 근처에도 놓지 못한다. 아래 수치는 수정 시점(f=3478.3)에 측정한 것이다. f=2318.8 에서는 시차항이 290px 로 줄지만 밴드 폭보다 여전히 한 자릿수 크므로 결론은 같다."));
body.push(Gap(60));
body.push(Tbl(
  ["V선", "실제 위치", "예측 (시차 보정)", "예측 (보정 없음)"],
  [
    ["V0", "141.3 px", { t: "150.8  (+9.5)", c: C.accent }, { t: "585.6  (+444.3)", c: C.warn }],
    ["V10", "1090.4 px", { t: "1099.6  (+9.3)", c: C.accent }, { t: "1534.4  (+444.1)", c: C.warn }],
    ["V20", "2087.2 px", { t: "2096.6  (+9.4)", c: C.accent }, { t: "2531.4  (+444.2)", c: C.warn }],
    ["평균 오차", "—", { t: "35.8 px", c: C.accent }, { t: "470.6 px", c: C.warn }],
  ],
  [1200, 2200, 3100, 3138]));

/* ================= 6. 입력 시험 ================= */
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("6. 현장 이미지 입력 시험"));

body.push(H2("6.1 시험 방법"));
body.push(P("현장 촬영 이미지(주황 격자, 471×372, 벽+바닥+기둥, 어두운 배경)의 성질을 그대로 재현한 입력을 만들어 A_선검출.detect()를 실제로 실행하였다. 비교군으로 동일 장면에 녹색 격자를 넣은 경우와, 레이저를 통째로 지운 경우를 함께 측정하였다."));

body.push(H2("6.2 결과"));
body.push(Tbl(
  ["입력", "반환 선 수", "화면 안", "실제 격자선까지 평균 거리"],
  [
    ["주황 격자", "21 / 21", "5", { t: "332.05 px", c: C.warn }],
    ["녹색 격자", "21 / 21", "5", { t: "332.03 px", c: C.warn }],
    ["레이저 없음 (장면만)", "21 / 21", "5", { t: "332.05 px", c: C.warn }],
  ],
  [3000, 2200, 1800, 2638]));
body.push(P("로그는 「[A] 완료(최종 통합): V=21/21 H=21/21」로 42개 선 전부 검출 성공으로 표시된다.", { before: 140 }));
body.push(Note("결정적 검증: 레이저를 통째로 지운 이미지와 주황 격자 이미지의 출력이 픽셀 단위로 완전히 동일(0.00px 차이)하다. 즉 출력이 입력 레이저와 아무 상관이 없으며, 검출값이 아니라 기하 예측값이 그대로 반환된 것이다."));

body.push(H2("6.3 원인"));
body.push(H3("① 레이저 색 — 주황은 신호로 잡히지 않음"));
body.push(P("A_선검출의 신호 분리는 G − (R+B)/2이며 녹색 레이저(설계 사양 520nm) 전용이다. 주황·빨강은 음수가 되어 0으로 잘린다. 실제로 주황 이미지에서 이 값의 최대치는 18.0이었는데, 그것은 레이저가 아니라 모래빛 바닥(196,176,120)의 색 편향이었다. 검출기가 바닥을 레이저로 오인하고 추적한 것이다."));
body.push(H3("② f_px 와 해상도 불일치"));
body.push(P("f_px 는 2448px 센서 기준값이다(시험 당시 3478.3, 현재 2318.8). 471px 이미지에 그대로 쓰면 예측 격자가 화면 밖으로 나가 21개 중 5개만 안에 들어왔다. 해상도에 맞춰 환산하면 전부 화면 안에 들어오고 평균 오차가 332px에서 7px로 떨어진다. 다만 7px도 격자 간격 23px의 30%라 측정에는 쓸 수 없다. 이 환산은 calibration.scale_to_resolution() 으로 제공한다."));
body.push(H3("③ 조용한 실패"));
body.push(P("신호가 없으면 _fallback_geom()이 기하 예측을 반환하고, _validate_and_fix()가 이상선을 인접선 보간값으로 교체한다. 그 결과 검출이 0개여도 출력은 항상 42개다. 실패가 성공처럼 보이고, 조작된 좌표가 그대로 삼각측량에 들어가 그럴듯한 각도가 산출된다. 현장에서 가장 위험한 동작이다."));

body.push(H2("6.4 필요 조치"));
body.push(Bul("원본 해상도 이미지 — 리사이즈된 사진은 f_px가 같은 비율로 줄어야 하고, 서브픽셀 정밀도(σ_u≈0.2px)가 원본 기준이라 축소분만큼 손실된다."));
body.push(Bul("레이저 색에 맞는 신호 분리 — 주황이면 R − (G+B)/2, 또는 색과 무관한 ON/OFF 차영상(PDF 2.2에 이미 하드웨어 사양으로 존재)."));
body.push(Bul("해당 장비의 캘리브레이션 값 — 4.2절의 f, b, c_x·c_y, α_i, β_j."));
body.push(Bul("IMU 중력벡터 — 없으면 수직·수평도 자체가 정의되지 않는다."));
body.push(Bul("검출 실패를 거부로 처리 — detect()가 실제 검출률을 반환하고, 임계 미만이면 기하 예측을 채우는 대신 거부하도록 변경."));

/* ================= 7. 검증 ================= */
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("7. 검증 결과"));

body.push(H2("7.1 회귀 검증 (tests/test_regression.py — 9군 전체 통과)"));
body.push(Tbl(
  ["검증 항목", "결과"],
  [
    ["eq1 삼각측량 복원오차 (0.5 / 1.0 / 1.5 / 2.0 m)", "0.000000 mm"],
    ["eq3 v1 규약 하위호환 (수직도 / 수평도)", "오차 < 1e-6 °"],
    ["eq3 장비 37° 기울임 불변성", "오차 < 1e-6 °"],
    ["중력 두 경로 일치 (IMU vs 카메라 자세, pitch 0/22/90°)", "오차 0"],
    ["eq2 TLS 평면적합 — 경사면", "0.0013° (기존 방식 8.45°)"],
    ["eq2 TLS 평면적합 — 정면 벽 회귀 여부", "0.0000° 차이 (회귀 없음)"],
    ["eq2 축 적합 (0 / 0.6 / 1.2 / 3.0°)", "오차 < 0.03 °"],
    ["eq5 경계 정제 (깊이 불연속 · 마스크 침식)", "정상"],
    ["하드웨어 사양 정합성 (pdf / improved 두 프로파일)", "격자 수용 · 심도 · σ_Z 목표 모두 만족"],
    ["가림 그림자로 벽이 쪼개지지 않음", "벽 영역 1개 (gt · geom)"],
  ],
  [6800, 2838]));

body.push(H2("7.2 합성 씬 정답 대조"));
body.push(P("벽 + 바닥 + 동바리가 한 프레임에 들어오는 합성 씬에서 두 세그멘테이션 백엔드 모두 전 항목 통과하였다. 장비를 34° 하향으로 잡아 벽과 바닥을 한 장에 담았다. 개선 프로파일에서 V선 위 격자점은 9,537개(벽 6,896 · 바닥 2,032 · 동바리 609)이며, PDF 원안에서는 4,826개(벽 3,503 · 바닥 1,022 · 동바리 301)다."));
body.push(Gap(60));
body.push(Tbl(
  ["부재", "검측 항목", "정답", "PDF 원안", "개선안 (gt)", "개선안 (geom)"],
  [
    ["벽", "수직도", "0.50 °", "오차 0.0150 °", { t: "오차 0.0135 °", c: C.accent }, "오차 0.0135 °"],
    ["바닥", "수평도", "0.30 °", "오차 0.0176 °", { t: "오차 0.0005 °", c: C.accent }, "오차 0.0056 °"],
    ["동바리", "축 수직도", "1.20 °", "오차 0.1431 °", { t: "오차 0.0239 °", c: C.accent }, "오차 0.0569 °"],
    ["벽", "직선자 처짐", "3.99 mm", "4.76 mm", { t: "3.91 mm", c: C.accent }, "3.91 mm"],
    ["벽", "요철 깊이", "융기 6.0 mm", "5.57 mm", "5.17 mm", "5.17 mm"],
  ],
  [900, 1500, 1500, 1500, 1600, 1638]));
body.push(P("허용 기준은 ±0.5°이다. 직선자 처짐의 정답 3.99mm 는 융기 높이 6.0mm 가 아니라 그 융기 위에 자를 얹었을 때 생기는 틈이다(8.1 참조). 요철 깊이가 20% 작게 나오는 것은 융기가 평면 적합을 끌어올리는 소프트웨어 편향으로, 사양으로는 개선되지 않는다.", { before: 100, size: 18, color: C.ink2 }));

body.push(H3("동바리 — 노출 길이가 정확도를 지배한다"));
body.push(P("세로 시야는 1.2m 에서 1,060mm 다. 2.4m 부재 중 일부만 담기며, 짧게 담길수록 원통 단면이 주축을 끌어당겨 각도가 흔들린다. Ø48.6mm 파이프서포트로 측정한 값이다."));
body.push(Gap(60));
body.push(Tbl(
  ["보이는 축 길이", "세장비 L/r", "최대 오차 (20회 시행)"],
  [
    ["100 mm", "4.1", { t: "11.57 °", c: C.warn }],
    ["178 mm", "7.3", { t: "2.08 °", c: C.warn }],
    ["300 mm", "12.3", { t: "1.82 °", c: C.warn }],
    ["500 mm", "20.6", "0.64 °"],
    ["800 mm", "32.9", { t: "0.18 °", c: C.accent }],
    ["1200 mm", "49.4", { t: "0.06 °", c: C.accent }],
  ],
  [3200, 2800, 3638]));
body.push(P("경험식으로 각도 불확실도 ≈ 13 / (L/r) 이며, 허용 ±0.5° 를 지키려면 L/r ≥ 26 (Ø48.6mm 기준 약 0.63m)이 필요하다. 코드는 L/r < 13 을 하드 기각하고, 그 이상은 불확실도를 판정에 반영해 측정값에 그 폭을 더했을 때 허용치를 넘으면 「판정보류(노출길이)」로 내보낸다.", { before: 140 }));

body.push(H2("7.3 세그멘테이션 오차 분해"));
body.push(P("Isaac Semantics가 화소별 정답 마스크를 제공하므로, 세그멘테이션 오차만 0으로 만든 상태를 실제로 구성할 수 있다. 이를 기준선으로 두고 마스크를 훼손해가며 최종 오차 중 세그멘테이션이 더하는 몫을 분리 측정하였다."));
body.push(Gap(60));
body.push(Tbl(
  ["조건", "벽", "바닥", "동바리", "판정"],
  [
    ["정답 마스크 (기준선)", "0.0150", "0.0176", "0.1431", "통과"],
    ["마스크 +2px 팽창", "0.0153", "0.0176", "0.2403", "통과"],
    ["마스크 +8px 팽창", "0.0165", "0.0175", "0.2403", "통과"],
    ["마스크 +16px 팽창", "0.0172", "0.0178", "0.2246", "통과"],
    ["라벨 오분류 9종", "복구", "복구", { t: "8/9 복구", c: C.warn }, "통과"],
    ["기하 전용 백엔드 (geom)", "0.0274", "0.0244", "0.2584", "통과"],
  ],
  [3400, 1500, 1500, 1500, 1738]));
body.push(P("단위는 도(°)이며 허용 기준은 ±0.5°이다.", { before: 100, size: 18, color: C.ink2 }));
body.push(Note("동바리 오차가 이전 표(0.02°대)보다 커진 것은 알고리즘이 나빠져서가 아니라 격자가 실사양이 되었기 때문이다. DOE 42.61°·20선이면 1.2m 격자 피치가 49.3mm 인데 동바리 지름은 Ø48.6mm 라, 부재 하나에 V선이 한두 개밖에 걸리지 않는다. 그래도 0.26° 로 허용 ±0.5° 안이다. 라벨이 67% 넘게 뒤섞인 한 시행에서는 그 몇 개마저 흩어져 축적합이 성립하지 않았고, 이때 알고리즘은 틀린 값을 내는 대신 측정을 포기했다. 얇은 부재를 안정적으로 재려면 촬영거리를 1.0m 쪽으로 당기거나(피치 41mm) 부재를 화면 중앙에 두는 운용이 필요하다."));
body.push(P("각도가 강건한 이유는 네 겹의 방어 때문이다.", { before: 140 }));
body.push(Bul("평면·축 적합이 수천 점을 평균하므로 경계 몇 px는 묻힌다."));
body.push(Bul("의미×기하 융합이 벽/바닥 혼동을 기하로 되돌린다."));
body.push(Bul("병합된 영역을 기하 부정합 검사로 되쪼갠다."));
body.push(Bul("얇은 부재는 robust 축 적합으로 실루엣 오염을 걷어낸다."));
body.push(P("세그멘테이션은 「무엇을 재야 하는지」를 정하고, 「얼마인지」는 기하가 결정하기 때문이다. 다만 다음 둘은 세그멘테이션 품질이 그대로 결과가 된다.", { before: 140 }));
body.push(Bul("부재 종류 구분 — 동바리/기둥/철근, 벽/거푸집/조적은 기하가 구분하지 못한다. 틀리면 각도는 맞아도 KCS 허용치를 잘못 적용한다."));
body.push(Bul("평활도 — 영역 경계와 점 밀도에 직접 좌우된다. σ_u가 0.5px를 넘으면 법선방향 불확실도가 목표 2mm를 넘어 판정이 보류된다."));

/* ================= 8. 한계 ================= */
body.push(H1("8. 한계 및 후속 과제"));

body.push(H2("8.1 확인된 한계"));
body.push(Tbl(
  ["항목", "내용"],
  [
    ["평활도 참값 비교 정정", "이전 판은 「직선자 처짐이 정답의 60~70% 로 과소보고된다」고 적었으나, 이는 융기 높이 6.0mm 와 비교한 탓이다. 직선자 처짐의 참값은 융기 높이가 아니라 gap(d) = A(1−d/D) − A·exp(−d²/2σ²) 의 최대값이며, 이 씬(D=430mm)에서 3.99mm 다. 무잡음·초고밀도 극한에서 파이프라인이 내는 값도 3.99mm 로 해석값과 일치한다. 즉 계통 편향은 없었다. 현재 오차는 PDF 원안 +0.54mm, 개선안 +0.08mm 다."],
    ["요철 깊이 과소보고", "eq4 의 최대잔차는 융기 높이보다 약 20% 작게 나온다. 융기가 평면 적합을 위로 끌어올리기 때문이며, 사양을 바꿔도 개선되지 않는 소프트웨어 편향이다. 요철 영역을 뺀 뒤 평면을 다시 적합하는 반복이 필요하다."],
    ["KCS 3m 기준 미달성", "1.5m 에서 격자가 덮는 폭이 1.17m 라 3m 직선자 기준(벽체 3m당 7mm, 천장 3m당 3mm)을 한 장으로 적용할 수 없다. 여러 장을 이어붙이거나 1m 기준으로 대체해야 한다. 화각을 넓히면 σ_Z 가 나빠지므로 사양으로 풀 문제가 아니다."],
    ["한 장의 한계", "벽 평활도(정면 촬영 유리)와 바닥 수평도(하향 촬영 필요)는 한 장으로 동시에 최적화할 수 없다. 정밀 평활도가 필요하면 면별로 정면 촬영을 따로 해야 한다."],
    ["Isaac 렌더 경로 미검증", "어노테이터 부착, 카메라 하향 자세, 선검출 실제 연동은 Isaac 환경이 없어 실행 검증하지 못했다. 계산 로직은 전부 Isaac 비의존 모듈에 있어 오프라인 검증되었다."],
    ["캘리브레이션 미확보", "4.2절 표에서 assumed 로 표시한 항목(c_x·c_y, α_i, β_j, R·t, R_ic, b_a)과 σ_u 가 가정값이다. PDF 2.2 에서 그대로 오는 값은 기선 150mm, DOE 42.61°, 격자 20+20 이고, 초점거리 8mm 는 나머지 사양에서 유도한 설계값이다."],
    ["렌즈 초점거리 확정 필요", "PDF 는 초점거리를 명시하지 않는다. 이 코드는 격자 수용·고정초점 심도·목표 정밀도 세 조건에서 8mm 로 정했다(5.3). 실제 조달 렌즈가 8mm 가 아니면 격자가 센서를 벗어나거나 심도가 작업거리를 못 덮는다. 발주 전 확정이 필요하다."],
    ["기구 설계 반영 필요", "레이저 축 6.18° 수렴은 기선 시차를 상쇄하려고 새로 넣은 설계값이다. 조사기 마운트에 이 각도가 반영되어야 하며, 실제 조립각을 캘리브레이션으로 다시 측정해야 한다."],
    ["카메라 항 수치 불일치", "PDF 2.2 의 「1.2m·FOV 200mm 에서 ≈0.08mm/px」는 초점거리 약 50mm 에 해당해, 같은 표의 DOE 936mm 와 양립하지 않는다. 근접 확대 촬영을 상정한 별도 수치로 보고 반영하지 않았다. 확인이 필요하다."],
    ["얇은 부재 표본 부족", "1.2m 격자 피치 49.3mm 는 Ø48.6mm 동바리와 거의 같다. 부재 하나에 V선이 한두 개만 걸려 축 방향 표본이 얕다. 현재 오차 0.26° 로 기준은 만족하지만 여유가 크지 않다."],
    ["세그멘테이션 백엔드", "sam(GroundingDINO+SAM2)·vlm 백엔드는 인터페이스만 있고 미구현이다. GPU와 모델 가중치가 필요해 검증 환경을 확보하지 못했다."],
  ],
  [2400, 7238]));

body.push(H2("8.2 후속 과제"));
body.push(Bul("발사각 α_i 실측 캘리브레이션 — 평면 타깃을 알려진 거리 2점에 두고 각 선의 발사각을 역산하는 절차 및 코드."));
body.push(Bul("선검출 실패의 명시적 거부 — detect()가 검출률을 반환하고 임계 미만이면 기하 예측 대체 없이 거부."));
body.push(Bul("색 무관 신호 분리 및 해상도 자동 환산 — camera_params의 resolution과 입력 이미지 크기가 다르면 f, c_x, c_y를 자동 스케일."));
body.push(Bul("Phase 3 — SAM2 + GroundingDINO 백엔드 구현 및 VLM 라벨링."));
body.push(Bul("Isaac 환경에서 씬 생성 → 촬영 → 영역별 검측 전 구간 실행 검증."));
body.push(Bul("하드웨어 프로토타입 제작 후 실제 콘크리트 환경 현장 검증."));

/* ================= 문서 ================= */
const doc = new Document({
  creator: "광운대학교 건설시스템공학 연구실",
  title: "레이저 그리드 기반 영역별 품질검측 파이프라인",
  description: "구조 · 파라미터 · 입력 데이터 정리",
  styles: {
    default: {
      document: { run: { font: FONT, size: 20, color: C.ink } },
    },
  },
  numbering: {
    config: [{
      reference: "bul",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 240 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 960, hanging: 240 } } } },
      ],
    }],
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 },
      },
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("레이저그리드_영역별품질검측_파이프라인_정리.docx", buf);
  console.log("written:", buf.length, "bytes");
});

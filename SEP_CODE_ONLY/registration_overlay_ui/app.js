(() => {
  "use strict";

  const data = window.CASES;
  if (!data) throw new Error("data.js yüklenemedi");

  const ORGAN_LABEL = { aorta: "Aorta", spleen: "Spleen" };
  const PLANE_LABEL = { axial: "Aksiyel", coronal: "Koronal" };
  const state = {
    view: "gallery",
    caseId: data.cases[0].id,
    organ: "aorta",
    plane: "axial",
    sliceMode: "all",
    sliceIndex: 1,
    allIndex: 0,
    mode: "slider",
    split: 50,
    galleryQuery: "",
    gallerySort: "date",
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  const refs = {
    fixedRef: $("#fixedRef"),
    statGain: $("#statGain"),
    statGainCaption: $("#statGainCaption"),
    statLow: $("#statLow"),
    statLowCaption: $("#statLowCaption"),
    statCount: $("#statCount"),
    galleryView: $("#galleryView"),
    stats: $(".stats"),
    galleryTotal: $("#galleryTotal"),
    galleryGrid: $("#galleryGrid"),
    galleryCount: $("#galleryCount"),
    galleryFilterNote: $("#galleryFilterNote"),
    gallerySearch: $("#gallerySearch"),
    gallerySort: $("#gallerySort"),
    galleryEmpty: $("#galleryEmpty"),
    detailView: $("#detailView"),
    detailCurrent: $("#detailCurrent"),
    backToGallery: $("#backToGallery"),
    caseTabs: $("#caseTabs"),
    caseTitle: $("#caseTitle"),
    caseSeries: $("#caseSeries"),
    caseTags: $("#caseTags"),
    caseNote: $("#caseNote"),
    organTabs: $("#organTabs"),
    planeTabs: $("#planeTabs"),
    planeHint: $("#planeHint"),
    sliceTabs: $("#sliceTabs"),
    sliceModeRow: $("#sliceModeRow"),
    sliceModeToggle: $("#sliceModeToggle"),
    allSliceControl: $("#allSliceControl"),
    allSliceRange: $("#allSliceRange"),
    allSliceLabel: $("#allSliceLabel"),
    allSliceCount: $("#allSliceCount"),
    allSliceMin: $("#allSliceMin"),
    allSliceMax: $("#allSliceMax"),
    viewEyebrow: $("#viewEyebrow"),
    viewLabel: $("#viewLabel"),
    viewMeta: $("#viewMeta"),
    compare: $("#compare"),
    afterPane: $("#afterPane"),
    compareDivider: $("#compareDivider"),
    imgBefore: $("#imgBefore"),
    imgAfter: $("#imgAfter"),
    splitRange: $("#splitRange"),
    sliderRow: $("#sliderRow"),
    modeToggle: $("#modeToggle"),
    metricRows: $("#metricRows"),
    methodText: $("#methodText"),
    insightText: $("#insightText"),
    limitText: $("#limitText"),
    caveatText: $("#caveatText"),
    caveatCard: $("#caveatCard"),
    printButton: $("#printButton"),
  };

  const getCase = () => data.cases.find((item) => item.id === state.caseId) ?? data.cases[0];
  const fixedDay = data.fixed.split("_")[0];
  const isSameDay = (item) => item.day === fixedDay;
  const displayTag = (item) => (
    !item.featured && !item.qc && isSameDay(item) ? "Aynı gün" : item.tag
  );
  const displayTone = (item) => (
    !item.featured && !item.qc && isSameDay(item) ? "sky" : item.tone
  );
  const getOrganViews = () => {
    const item = getCase();
    return item.views[state.organ] ?? item.views.aorta;
  };
  const hasAllSlices = () => (getOrganViews().all?.length ?? 0) > 1;
  const isAllMode = () => state.plane === "axial" && state.sliceMode === "all" && hasAllSlices();
  const getViewList = () => {
    const views = getOrganViews();
    if (isAllMode()) return views.all;
    return views[state.plane] ?? views.axial;
  };
  const activeIndex = () => {
    const list = getViewList();
    const index = isAllMode() ? state.allIndex : state.sliceIndex;
    return Math.max(0, Math.min(index, list.length - 1));
  };
  const currentView = () => {
    const list = getViewList();
    return list[activeIndex()] ?? list[0];
  };
  const middleAllIndex = () => {
    const all = getOrganViews().all ?? getOrganViews().axial;
    return Math.floor((all.length - 1) / 2);
  };
  const resetSlicePositions = () => {
    state.sliceIndex = state.plane === "coronal" ? 0 : 1;
    state.allIndex = middleAllIndex();
  };

  function scoreTone(item) {
    return item.qc ? "amber" : item.tone;
  }

  function pct(value) {
    return Math.max(0, Math.min(100, ((value - 0.75) / 0.25) * 100));
  }

  function formatScore(value) {
    return Number(value).toFixed(3);
  }

  function formatDelta(value) {
    const sign = value >= 0 ? "+" : "−";
    return `${sign}${Math.abs(value).toFixed(3)}`;
  }

  function renderStats() {
    let best = { delta: -Infinity, organ: "aorta", item: null };
    let low = { after: Infinity, organ: "aorta", item: null };
    data.cases.filter((item) => !item.qc).forEach((item) => {
      ["aorta", "spleen"].forEach((organ) => {
        const scores = item.scores[organ];
        const delta = scores.after - scores.before;
        if (delta > best.delta) best = { delta, organ, item };
        if (scores.after < low.after) low = { after: scores.after, organ, item };
      });
    });
    refs.statGain.textContent = `+${best.delta.toFixed(3)}`;
    refs.statGainCaption.textContent = `${ORGAN_LABEL[best.organ]} · ${best.item.series}`;
    refs.statLow.textContent = formatScore(low.after);
    refs.statLowCaption.textContent = `${ORGAN_LABEL[low.organ]} · ${low.item.series}`;
    refs.statCount.textContent = String(data.cases.length);
    refs.galleryTotal.textContent = String(data.cases.length);
    refs.fixedRef.textContent = data.fixed;
  }

  function renderCaseTabs() {
    refs.caseTabs.innerHTML = "";
    const featured = data.cases.filter((item) => item.featured);
    const selected = getCase();
    const items = selected.featured ? featured : [selected, ...featured];
    items.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "case-tab";
      button.dataset.caseId = item.id;
      button.dataset.tone = item.tone;
      button.innerHTML = `
        <div class="tab-kicker">
          <span class="tab-title">${item.title}</span>
          <span class="tab-badge">${displayTag(item)}</span>
        </div>
        <div class="tab-series">${item.series} · ${data.fixed}</div>
        <span class="tab-bar"></span>
      `;
      button.addEventListener("click", () => {
        if (state.caseId === item.id) return;
        state.caseId = item.id;
        state.organ = "aorta";
        state.plane = "axial";
        state.sliceMode = hasAllSlices() ? "all" : "samples";
        resetSlicePositions();
        render();
      });
      refs.caseTabs.appendChild(button);
    });
  }

  function galleryCases() {
    const query = state.galleryQuery.trim().toLowerCase();
    let items = data.cases.filter((item) => {
      if (!query) return true;
      return [item.series, item.day, String(item.take), item.title, item.tag, displayTag(item)]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
    const averageGain = (item) => (
      (item.scores.aorta.after - item.scores.aorta.before)
      + (item.scores.spleen.after - item.scores.spleen.before)
    ) / 2;
    const minFinal = (item) => Math.min(item.scores.aorta.after, item.scores.spleen.after);
    const sorters = {
      date: (a, b) => a.order - b.order,
      gain: (a, b) => averageGain(b) - averageGain(a),
      aortaFinal: (a, b) => b.scores.aorta.after - a.scores.aorta.after,
      spleenFinal: (a, b) => b.scores.spleen.after - a.scores.spleen.after,
      low: (a, b) => minFinal(a) - minFinal(b),
    };
    return items.sort(sorters[state.gallerySort] ?? sorters.date);
  }

  function renderGallery() {
    const items = galleryCases();
    const total = data.cases.length;
    refs.galleryGrid.innerHTML = items.map((item) => {
      const aorta = item.scores.aorta;
      const spleen = item.scores.spleen;
      return `
        <button class="take-card${item.featured ? " featured" : ""}${item.qc ? " qc" : ""}" type="button"
                data-case-id="${item.id}" aria-label="${item.series} detayını aç">
          <div class="take-head">
            <div class="take-id">
              <span class="take-series">${item.series}</span>
              <span class="take-date">${item.day} · take ${item.take}${item.qc ? " · QC" : ""}</span>
            </div>
            <span class="take-tag" data-tone="${displayTone(item)}">${displayTag(item)}</span>
          </div>
          <div class="take-composite">
            <img src="${item.gallery}" alt="${item.series} · Aorta ve Spleen önce/sonra">
          </div>
          <div class="take-scores">
            <div class="take-score">
              <span>A</span><b>${formatScore(aorta.before)}</b><i>→</i><b>${formatScore(aorta.after)}</b>
              <em class="${aorta.after >= aorta.before ? "pos" : "neg"}">${formatDelta(aorta.after - aorta.before)}</em>
            </div>
            <div class="take-score">
              <span>S</span><b>${formatScore(spleen.before)}</b><i>→</i><b>${formatScore(spleen.after)}</b>
              <em class="${spleen.after >= spleen.before ? "pos" : "neg"}">${formatDelta(spleen.after - spleen.before)}</em>
            </div>
          </div>
        </button>
      `;
    }).join("");
    refs.galleryCount.textContent = `${items.length} / ${total} take`;
    refs.galleryFilterNote.textContent = state.galleryQuery
      ? `Arama: “${state.galleryQuery}”`
      : state.gallerySort === "date" ? "Tümü gösteriliyor" : "Sıralı görünüm";
    refs.galleryEmpty.hidden = items.length > 0;
    refs.galleryEmpty.textContent = "Bu aramaya uyan take yok.";
  }

  function openCase(caseId) {
    state.view = "detail";
    state.caseId = caseId;
    state.organ = "aorta";
    state.plane = "axial";
    state.sliceMode = hasAllSlices() ? "all" : "samples";
    resetSlicePositions();
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showGallery() {
    state.view = "gallery";
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderSidebar() {
    const item = getCase();
    refs.caseTitle.textContent = item.title;
    refs.caseSeries.textContent = `${item.series} → ${data.fixed}`;
    refs.caseNote.textContent = item.note;
    refs.caseTags.innerHTML = `
      <span class="pill tone">${displayTag(item)}</span>
      <span class="pill">${ORGAN_LABEL.aorta} + ${ORGAN_LABEL.spleen}</span>
      <span class="pill">fixed ${data.fixed}</span>
    `;

    refs.organTabs.innerHTML = "";
    Object.entries(ORGAN_LABEL).forEach(([organ, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.className = organ === state.organ ? "active" : "";
      button.setAttribute("aria-selected", String(organ === state.organ));
      button.addEventListener("click", () => {
        state.organ = organ;
        resetSlicePositions();
        render();
      });
      refs.organTabs.appendChild(button);
    });

    refs.planeTabs.innerHTML = "";
    Object.entries(PLANE_LABEL).forEach(([plane, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.className = plane === state.plane ? "active" : "";
      button.setAttribute("aria-selected", String(plane === state.plane));
      button.addEventListener("click", () => {
        state.plane = plane;
        resetSlicePositions();
        render();
      });
      refs.planeTabs.appendChild(button);
    });

    const organViews = getOrganViews();
    const allViews = organViews.all ?? [];
    const showSliceMode = state.plane === "axial" && allViews.length > 1;
    const views = getViewList();

    refs.sliceModeRow.hidden = !showSliceMode;
    $$("#sliceModeToggle button").forEach((button) => {
      button.classList.toggle("active", button.dataset.sliceMode === state.sliceMode);
    });

    refs.sliceTabs.innerHTML = "";
    refs.sliceTabs.hidden = isAllMode();
    refs.allSliceControl.hidden = !isAllMode();

    if (isAllMode()) {
      const index = activeIndex();
      refs.planeHint.textContent = `${allViews.length} aksiyel kesit · tümü`;
      refs.allSliceRange.max = String(allViews.length - 1);
      refs.allSliceRange.value = String(index);
      refs.allSliceLabel.textContent = allViews[index].label;
      refs.allSliceCount.textContent = `${allViews.length} kesit`;
      refs.allSliceMin.textContent = allViews[0].label;
      refs.allSliceMax.textContent = allViews[allViews.length - 1].label;
    } else {
      refs.planeHint.textContent = state.plane === "coronal"
        ? "1 koronal kesit"
        : `${views.length} örnek aksiyel kesit`;
      views.forEach((view, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `slice-button${index === state.sliceIndex ? " active" : ""}`;
        button.innerHTML = `<span>${view.label}</span><span class="slice-index">kesit ${index + 1}</span>`;
        button.addEventListener("click", () => {
          state.sliceIndex = index;
          render();
        });
        refs.sliceTabs.appendChild(button);
      });
    }
  }

  function renderViewer() {
    const item = getCase();
    const list = getViewList();
    const index = activeIndex();
    const view = list[index] ?? list[0];
    const organLabel = ORGAN_LABEL[state.organ];
    refs.viewEyebrow.textContent = isAllMode()
      ? "Tüm kesitler"
      : state.mode === "side" ? "Yan yana" : "Önce / sonra";
    refs.viewLabel.textContent = `${organLabel} · ${PLANE_LABEL[state.plane].toLowerCase()} · ${view.label}`;
    refs.viewMeta.textContent = `${data.fixed} fixed · ${item.series} moving`;
    refs.imgBefore.src = view.before;
    refs.imgAfter.src = view.after;
    refs.imgBefore.alt = `${item.series} · ${organLabel} · ${view.label} · önce`;
    refs.imgAfter.alt = `${item.series} · ${organLabel} · ${view.label} · sonra`;

    if (isAllMode()) {
      refs.allSliceRange.value = String(index);
      refs.allSliceLabel.textContent = view.label;
    }

    // Preload neighboring slices so scrubbing through the full stack stays smooth.
    [index - 1, index, index + 1]
      .filter((i) => i >= 0 && i < list.length)
      .forEach((i) => {
        [list[i].before, list[i].after].forEach((src) => {
          const img = new Image();
          img.src = src;
        });
      });

    $$(".case-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.caseId === state.caseId);
    });
    $$("#organTabs button").forEach((button) => {
      button.classList.toggle("active", button.textContent === organLabel);
    });
    $$("#planeTabs button").forEach((button) => {
      button.classList.toggle("active", button.textContent === PLANE_LABEL[state.plane]);
    });
    $$(".slice-button").forEach((button, index) => {
      button.classList.toggle("active", index === state.sliceIndex);
    });
    $$("#modeToggle button").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === state.mode);
    });

    refs.compare.classList.toggle("slider", state.mode === "slider");
    refs.compare.classList.toggle("side", state.mode === "side");
    refs.compareDivider.style.display = state.mode === "side" ? "none" : "";
    refs.splitRange.disabled = state.mode === "side";
    refs.sliderRow.style.display = state.mode === "side" ? "none" : "";
  }

  function averageGainOf(item) {
    return (
      (item.scores.aorta.after - item.scores.aorta.before)
      + (item.scores.spleen.after - item.scores.spleen.before)
    ) / 2;
  }

  function limitingOrgan(item) {
    return item.scores.aorta.after <= item.scores.spleen.after ? "aorta" : "spleen";
  }

  function afterPrefix(item) {
    return item.registration?.method === "identity" ? "Aynı geometri" : "Rigid sonrası";
  }

  function residualSentence(item, organ) {
    const residual = item.residual?.[organ];
    if (!residual) return "";
    return `${afterPrefix(item)}: ${ORGAN_LABEL[organ]} centroid farkı ${residual.centroid.toFixed(2)} mm, `
      + `HD95 ${residual.hd95.toFixed(2)} mm, ASSD ${residual.assd.toFixed(2)} mm.`;
  }

  function explainLimit(item) {
    const organ = limitingOrgan(item);
    const label = ORGAN_LABEL[organ];
    const final = item.scores[organ].after;
    const gain = averageGainOf(item);
    const residual = item.residual?.[organ];
    const residualText = residualSentence(item, organ);
    const stem = `Sınırlayan organ ${label}; final Dice ${formatScore(final)}.`;

    if (item.qc) {
      if (item.qcReason === "recon_grid_anomaly") {
        return `${stem} Bu pair recon grid anomalisi nedeniyle QC dışı. ${residualText} `
          + "Global hizalama hâlâ belirgin şekilde bozuk; farklı recon geometrisi/interpolasyon da hesaba giriyor. "
          + "Rigid’in tek başına bunu kapatması beklenmez; önce recon ve segmentasyon doğrulanmalı.";
      }
      if (item.qcReason === "low_label_artifact") {
        return `${stem} Bu take düşük label artifact’i taşıyor; ${label.toLowerCase()} kontur sınırı güvenilir değil. `
          + "Dice’daki düşüklük gerçek anatomik uyumsuzluktan çok segmentasyon artifact’ini yansıtıyor. "
          + "Daha yüksek Dice için registration değil, segmentasyon düzeltmesi gerekir.";
      }
      return `${stem} Bu take QC işaretli (${item.qcReason}); Dice yorumu temkinli yapılmalı. `
        + "Düşük değer registration’dan çok segmentasyon/recon kaynaklı olabilir.";
    }

    if (!item.inWorkbook) {
      return `${stem} ${residualText} Bu take workbook dışı bırakılmış; yani protokol dışı bir örnek. `
        + "Global hizalama iyi olsa da kalan fark boundary/şekil kaynaklı ve bu değerlendirme yalnızca rigid sonucu gösteriyor.";
    }

    if (isSameDay(item)) {
      const extra = Math.abs(gain) < 0.005
        ? "rigid’in ek global katkısı yok."
        : "rigid’in ek global katkısı sınırlı.";
      return `${stem} ${residualText} Aynı fixed günü: global geometri zaten çok yakın; ${extra} `
        + "Kalan fark ağırlıklı olarak segmentasyon/partial volume kaynaklı. "
        + "Daha yüksek Dice için yeni bir rigid denemesi değil, segmentasyon sınırının düzeltilmesi gerekir.";
    }

    if (item.id === "low") {
      return `${stem} ${residualText} 0731 cohort’u boundary/shape outlier: centroid farkı küçük olsa da `
        + "Aorta/Spleen konturlarında lokal şekil farkı kalıyor. Rigid yalnızca öteleme+dönme yaptığı için "
        + "bu kısmı düzeltemez; daha yüksek Dice için deformable registration veya segmentasyon düzeltmesi gerekir.";
    }

    const globalAligned = residual ? residual.centroid <= 1 && residual.hd95 <= 1.5 : true;
    if (globalAligned) {
      return `${stem} ${residualText} Centroid farkı 1 mm altında; yani kontur global olarak oturmuş. `
        + `Kalan problem görüntüde global kayma değil, ${label.toLowerCase()} kenarındaki şekil/segmentasyon farkı. `
        + (organ === "aorta"
          ? "Aorta küçük ve kıvrımlı olduğu için aynı mm kenar hatası Dice’ı Spleen’den daha çok düşürür. "
          : "")
        + "Rigid sadece global hizalamayı düzeltir; lokal şekil farkını kapatamaz. "
        + "Daha yüksek Dice için deformable registration veya segmentasyon düzeltmesi gerekir.";
    }

    return `${stem} ${residualText} Global hizalama 1 mm üzerinde kalmış; bu nedenle Dice’ı sınırlayan şey yalnızca kenar değil, `
      + "kalan centroid/rotasyon farkı da olabilir. Bu vakada önce transform ve segmentasyon QC’si kontrol edilmeli; "
      + "rigid’in tek başına yeterli olması beklenmez.";
  }

  function renderMetrics() {
    const item = getCase();
    refs.methodText.textContent = item.method;
    refs.metricRows.innerHTML = "";
    ["aorta", "spleen"].forEach((organ) => {
      const scores = item.scores[organ];
      const delta = scores.after - scores.before;
      const residual = item.residual?.[organ];
      const row = document.createElement("div");
      row.className = `metric-row${organ === state.organ ? " active" : ""}`;
      row.dataset.tone = scoreTone(item);
      row.innerHTML = `
        <div class="metric-top">
          <span class="metric-name">${ORGAN_LABEL[organ]}</span>
          <span class="metric-delta">${formatDelta(delta)}</span>
        </div>
        <div class="metric-values">
          <span class="metric-before">${formatScore(scores.before)}</span>
          <span class="metric-arrow">→</span>
          <span class="metric-after">${formatScore(scores.after)}</span>
        </div>
        <div class="metric-bar" aria-hidden="true">
          <span class="metric-bar-fill" style="width:${pct(scores.after)}%"></span>
          <span class="metric-before-marker" style="left:${pct(scores.before)}%"></span>
        </div>
        <div class="metric-axis"><span>0.75</span><span>1.00</span></div>
        ${residual ? `<div class="metric-residual">${afterPrefix(item)}: centroid ${residual.centroid.toFixed(2)} mm · HD95 ${residual.hd95.toFixed(2)} mm · ASSD ${residual.assd.toFixed(2)} mm</div>` : ""}
      `;
      refs.metricRows.appendChild(row);
    });

    const gainAorta = item.scores.aorta.after - item.scores.aorta.before;
    const gainSpleen = item.scores.spleen.after - item.scores.spleen.before;
    const averageGain = (gainAorta + gainSpleen) / 2;
    const minFinal = Math.min(item.scores.aorta.after, item.scores.spleen.after);
    const noChange = Math.abs(gainAorta) < 0.001 && Math.abs(gainSpleen) < 0.001;
    const isLow = item.qc || minFinal < 0.93 || (averageGain < 0.015 && !noChange);
    refs.insightText.textContent = item.qc
      ? `QC işaretli take (${item.qcReason}); metrikler ve overlay yine de gösteriliyor, yorumda dikkat.`
      : noChange
        ? "Aynı fixed günü / aynı geometri: centroid baseline ile rigid sonuç aynı; konturlar değişmiyor."
        : averageGain > 0.08
          ? "Belirgin kazanç: centroid kayması rigid ile büyük ölçüde düzeliyor; iki organda da örtüşme artıyor."
          : averageGain > 0.03
            ? "Orta düzey kazanç: rigid konturları toparlıyor; organdaki kalan fark şekil/boundary kaynaklı."
            : "Kazanç sınırlı: centroid zaten yakın. Kalan açık boundary/shape farkı; rigid ile daha fazlası gelmiyor.";
    refs.caveatText.textContent = item.qc
      ? `QC: ${item.qcReason}. Bu take workbook yorumunda ayrıca ele alınmalı; overlay yalnızca görsel kontrol içindir.`
      : !item.inWorkbook
        ? "Workbook dışı bırakılan take; kayıt yine de karşılaştırma için gösteriliyor."
        : item.id === "low"
          ? "0731 cohort’u boundary/shape outlier. Deformable veya segmentation düzeltmesi ayrı bir yöntem; bu değerlendirme rigid sonucu gösterir."
          : "Önce: centroid translation baseline. Sonra: üretim rigid transformu. Amaç rigid’in katkısını görünür kılmak.";
    refs.limitText.textContent = explainLimit(item);
    refs.caveatCard.style.borderColor = isLow ? "rgba(255, 190, 105, 0.18)" : "";
  }

  function setSplit(value) {
    state.split = Math.max(0, Math.min(100, Number(value)));
    refs.compare.style.setProperty("--split", `${state.split}%`);
    refs.splitRange.value = String(Math.round(state.split));
  }

  function render() {
    refs.galleryView.hidden = state.view !== "gallery";
    refs.detailView.hidden = state.view !== "detail";
    refs.stats.hidden = state.view === "detail";

    if (state.view === "gallery") {
      renderGallery();
      return;
    }

    const organViews = getOrganViews();
    const allViews = organViews.all ?? organViews.axial;
    state.sliceIndex = Math.max(0, Math.min(state.sliceIndex, organViews.axial.length - 1));
    state.allIndex = Math.max(0, Math.min(state.allIndex, allViews.length - 1));
    refs.detailCurrent.textContent = `${getCase().series} · fixed ${data.fixed}`;
    renderCaseTabs();
    renderSidebar();
    renderViewer();
    renderMetrics();
    setSplit(state.split);
  }

  refs.gallerySearch.addEventListener("input", (event) => {
    state.galleryQuery = event.target.value;
    renderGallery();
  });

  refs.gallerySort.addEventListener("change", (event) => {
    state.gallerySort = event.target.value;
    renderGallery();
  });

  refs.galleryGrid.addEventListener("click", (event) => {
    const card = event.target.closest(".take-card");
    if (card) openCase(card.dataset.caseId);
  });

  refs.backToGallery.addEventListener("click", showGallery);

  refs.modeToggle.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    state.mode = button.dataset.mode;
    renderViewer();
    setSplit(state.split);
  });

  refs.sliceModeToggle.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-slice-mode]");
    if (!button || state.plane !== "axial") return;
    state.sliceMode = button.dataset.sliceMode;
    if (state.sliceMode === "all") state.allIndex = middleAllIndex();
    render();
  });

  refs.allSliceRange.addEventListener("input", (event) => {
    state.allIndex = Number(event.target.value);
    renderViewer();
  });

  refs.splitRange.addEventListener("input", (event) => setSplit(event.target.value));

  let dragging = false;
  function updateFromPointer(event) {
    const rect = refs.compare.getBoundingClientRect();
    const value = ((event.clientX - rect.left) / rect.width) * 100;
    setSplit(value);
  }

  refs.compare.addEventListener("pointerdown", (event) => {
    if (state.mode !== "slider") return;
    dragging = true;
    refs.compare.setPointerCapture?.(event.pointerId);
    updateFromPointer(event);
  });
  refs.compare.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    updateFromPointer(event);
  });
  refs.compare.addEventListener("pointerup", () => { dragging = false; });
  refs.compare.addEventListener("pointercancel", () => { dragging = false; });

  document.addEventListener("keydown", (event) => {
    if (state.view !== "detail") return;
    if (event.target.closest("input, textarea, button")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const views = getViewList();
      const delta = event.key === "ArrowRight" ? 1 : -1;
      const key = isAllMode() ? "allIndex" : "sliceIndex";
      state[key] = Math.max(0, Math.min(views.length - 1, state[key] + delta));
      render();
      event.preventDefault();
    }
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      state.organ = state.organ === "aorta" ? "spleen" : "aorta";
      resetSlicePositions();
      render();
      event.preventDefault();
    }
  });

  refs.printButton.addEventListener("click", () => window.print());

  state.allIndex = middleAllIndex();
  renderStats();
  renderCaseTabs();
  render();
})();

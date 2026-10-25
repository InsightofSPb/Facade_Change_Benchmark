/**
 * Deterministic demonstration reporter. No network requests or model inference.
 * A future provider receives the same evidence object and returns a report.
 */
export async function buildReport(
  { facade, categories = [], location = null, reviewNotes = [] } = {},
  provider = null,
) {
  if (!facade || typeof facade !== 'object') {
    throw new TypeError('Для отчёта необходимы данные фасада.');
  }
  if (provider !== null) {
    if (typeof provider !== 'function') {
      throw new TypeError('Поставщик отчёта должен быть функцией.');
    }
    // Provider failures remain visible; never replace a failed real report with a stub.
    const report = await provider({ facade, categories, location, reviewNotes });
    if (!report || typeof report !== 'object' || Array.isArray(report)) {
      throw new TypeError('Поставщик отчёта вернул некорректный результат.');
    }
    return report;
  }

  const frames = Array.isArray(facade.frames) ? facade.frames : [];
  const pairs = Array.isArray(facade.pairs) ? facade.pairs : [];
  const byCategory = new Map(categories.map(category => [String(category.id), category]));
  const byFrame = new Map(frames.map((frame, index) => [frame.index ?? index, frame]));
  const facadeLabel = facade.label || facade.display_label || facade.name
    || facade.address || location?.address || facade.building_id || facade.id || 'Фасад';
  const photo = frame => frame?.crop_url || frame?.aligned_url || frame?.original_url || null;
  const observations = [];

  pairs.forEach((pair, pairIndex) => {
    const before = byFrame.get(pair.from_index);
    const after = byFrame.get(pair.to_index);
    const pairId = pair.id ?? pair.pair_id
      ?? `${facade.id || facade.building_id || 'facade'}:pair:${pairIndex}`;
    const events = Array.isArray(pair.events) ? pair.events : [];
    events.forEach((event, eventIndex) => {
      if (!Number.isFinite(event.area_pixels) || event.area_pixels <= 0) return;
      if (event.direction !== 'added' && event.direction !== 'removed') {
        throw new TypeError(`Неизвестное направление изменения в паре ${pairId}.`);
      }
      const category = byCategory.get(String(event.category_id));
      const categoryLabel = category?.label || category?.name || `Класс ${event.category_id}`;
      observations.push({
        from_year: pair.from_year ?? before?.year ?? null,
        to_year: pair.to_year ?? after?.year ?? null,
        category: categoryLabel,
        label: `${event.direction === 'added' ? 'Добавление' : 'Исчезновение'} метки: ${categoryLabel}`,
        direction: event.direction,
        area_pixels: event.area_pixels,
        bbox: Array.isArray(event.bbox) ? [...event.bbox] : null,
        pair_id: pairId,
        event_id: event.id ?? `${pairId}:event:${eventIndex}`,
        photo_before: photo(before),
        photo_after: photo(after),
      });
    });
  });

  const years = [...new Set(frames.map(frame => frame.year).filter(Number.isFinite))]
    .sort((left, right) => left - right);
  const period = years.length === 0 ? ''
    : years.length === 1 ? ` за ${years[0]} год`
      : ` за ${years[0]}–${years[years.length - 1]} годы`;
  const availability = `Наблюдений${period}: ${frames.length}. Пар для сравнения: ${pairs.length}.`;
  const summary = observations.length
    ? `${availability} Кандидатов различий GT-разметки: ${observations.length}. Для проверки доступны классы, участки и фотографии до и после изменения метки.`
    : `${availability} В переданных данных кандидаты различий GT-разметки не выделены. Это не означает, что состояние фасада оценено как безопасное.`;

  return {
    provider: 'stub',
    is_demo: true,
    title: `Предварительный журнал изменений — ${facadeLabel}`,
    summary,
    priority: observations.length ? {
      label: 'Проверить изменения разметки',
      reason: 'В предоставленных парах есть участки добавления или исчезновения меток GT. Требуется просмотр соответствующих фотографий.',
    } : {
      label: 'Сопоставить наблюдения',
      reason: 'В предоставленных данных нет выделенных кандидатов изменений; приоритет по состоянию фасада не установлен.',
    },
    people_property_risk: null,
    heritage_risk: null,
    scope_note: 'Фотографии показывают контекст фасада. Сводка различий относится к подготовленным размеченным участкам; её нельзя распространять на весь фасад. Координаты областей заданы в пикселях подготовленного участка.',
    observations,
    limitations: [
      'Отчёт сформирован демонстрационной заглушкой. Qwen и другие модели не запускались.',
      'Изменение метки GT является кандидатом для проверки, а не подтверждением появления повреждения, ремонта или замены элемента.',
      'Площадь указана в пикселях подготовленного участка. Области разных классов и пар могут пересекаться; их площади нельзя складывать как общую площадь повреждений.',
      'Совмещение снимков и условия видимости могут влиять на интерпретацию различий. Геометрическое перекрытие не гарантирует видимость поверхности.',
      'Угроза людям и имуществу, утрата ценности наследия и техническое состояние фасада в этом отчёте не оценены.',
      'Ссылки на ГОСТ указывают принятую основу структуры материалов. Этот демонстрационный отчёт не удостоверяет соответствие требованиям обследования.',
    ],
    standards: [
      { name: 'ГОСТ 31937-2024', clause: '5.1.10' },
      { name: 'ГОСТ Р 56198-2014', clause: '7.2.1' },
    ],
    assessment: null,
  };
}

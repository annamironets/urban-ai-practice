import base64
import html
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.io as pio


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sovetsky"
DATA_FILE = DATA_DIR / "sovetsky_pipeline_results.geojson"
OUTPUT_FILE = BASE_DIR / "dashboard.html"

LAND_USE_LABELS = {
    "residential": "Жилое",
    "business": "Деловое",
    "recreation": "Рекреационное",
    "industrial": "Промышленное",
    "transport": "Транспортное",
    "special": "Специальное",
    "agriculture": "Сельскохозяйственное",
}

COLUMN_LABELS = {
    "block_id": "ID квартала",
    "land_use_label": "Землепользование",
    "share": "Доля ведущей функции",
    "building_population_estimate": "Расчётное население, чел.",
    "population_density_per_ha": "Расчётная плотность населения, чел./га",
    "site_area": "Площадь квартала, м²",
    "site_area_ha": "Площадь квартала, га",
    "footprint_area": "Пятно зданий, м²",
    "build_floor_area": "Площадь этажей, м²",
    "living_area": "Жилая площадь, м²",
    "non_living_area": "Нежилая площадь, м²",
    "fsi": "FSI",
    "gsi": "GSI",
    "mxi": "MXI",
    "l": "L, этажность",
    "osr": "OSR",
    "share_living": "Доля жилой площади",
    "share_non_living": "Доля нежилой площади",
    "area_accessibility": "Пешеходная доступность, мин.",
    "provision_strong": "Школьная обеспеченность в пределах порога",
    "provision_weak": "Школьная обеспеченность с учётом поиска за порогом",
    "kindergarten_provision_strong": "Обеспеченность детсадами в пределах порога",
    "kindergarten_provision_weak": "Обеспеченность детсадами с учётом поиска за порогом",
    "population_scenario": "Сценарий населения",
    "population": "Население, чел.",
    "population_reference_date": "Дата / период оценки населения",
    "territory_scope": "Охват территории",
    "population_source": "Источник оценки населения",
    "matches_analysis_boundary": "Совпадает с границей анализа",
    "service_type": "Тип учреждения",
    "norm_places_per_1000": "Норматив мест на 1 000 жителей",
    "target_places": "Целевые места",
    "equivalent_facilities_at_minimum_capacity": "Эквивалент учреждений по минимальной вместимости",
    "observed_capacity": "Вместимость без автоподстановки (источник не проверен)",
    "modelled_capacity": "Модельная вместимость",
    "remaining_target_places": "Остаток до целевого числа мест",
    "name": "Учреждение",
    "capacity_source": "Источник вместимости",
    "capacity": "Вместимость, мест",
    "enrollment": "Число учащихся/воспитанников",
    "vacant_capacity": "Свободные места",
    "occupancy_rate": "Доля занятых мест",
    "capacity_is_imputed": "Вместимость модельная",
    "demand": "Расчётный спрос на школьные места",
    "demand_left": "Неудовлетворённый спрос на школьные места",
    "demand_within": "Спрос на школы в пределах порога",
    "demand_without": "Спрос на школы за пределами порога",
    "capacity_left": "Остаток школьной вместимости",
    "capacity_within": "Школьная вместимость в пределах порога",
    "capacity_without": "Школьная вместимость за пределами порога",
    "kindergarten_demand": "Расчётный спрос на места в детсадах",
    "kindergarten_capacity": "Вместимость детсадов в квартале",
    "kindergarten_demand_left": "Неудовлетворённый спрос на места в детсадах",
    "kindergarten_demand_within": "Спрос на детсады в пределах порога",
    "kindergarten_demand_without": "Спрос на детсады за пределами порога",
    "kindergarten_capacity_left": "Остаток вместимости детсадов",
    "kindergarten_capacity_within": "Вместимость детсадов в пределах порога",
    "kindergarten_capacity_without": "Вместимость детсадов за пределами порога",
    "development_status": "Статус застройки",
    "forest_zone_share": "Доля квартала в лесных полигонах",
    "forest_zone_data_source": "Источник лесных данных",
    "territory_class": "Класс территории",
    "spacematrix_morphotype": "Морфотип Spacematrix (код)",
    "spacematrix_morphotype_ru": "Морфотип Spacematrix",
    "strelka_morphotype": "Морфотип Стрелки (код)",
    "strelka_morphotype_ru": "Морфотип Стрелки",
}


def _land_use_label(value):
    if pd.isna(value):
        return "Не определено"
    key = str(value).rsplit(".", 1)[-1].strip().lower()
    return LAND_USE_LABELS.get(key, str(value))


def _format_number(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def _plot_html(figure, include_plotlyjs=False):
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False, "responsive": True},
    )


def _table_html(dataframe, columns, table_id):
    available = [column for column in columns if column in dataframe.columns]
    if not available or dataframe.empty:
        return '<p class="empty-state">Нет сохранённых данных для отображения.</p>'
    table = dataframe[available].copy()
    table = table.rename(columns={column: COLUMN_LABELS.get(column, column) for column in available})
    return table.to_html(
        index=False,
        escape=True,
        na_rep="—",
        classes="data-table",
        table_id=table_id,
        border=0,
        float_format=lambda value: _format_number(value, 3),
    )


def _add_map_layer(map_object, geodataframe, column, label, categorical=False, show=False):
    if column not in geodataframe.columns:
        return

    layer_data = geodataframe.loc[geodataframe.geometry.notna()].copy()
    palette = px.colors.qualitative.Safe
    missing_color = "#b8bec7"

    if categorical:
        categories = sorted(layer_data[column].dropna().astype(str).unique())
        color_by_category = {
            category: palette[index % len(palette)]
            for index, category in enumerate(categories)
        }

        def style_function(feature):
            value = feature["properties"].get(column)
            return {
                "fillColor": color_by_category.get(str(value), missing_color),
                "color": "#374151",
                "weight": 0.6,
                "fillOpacity": 0.72,
            }

    else:
        numeric_values = pd.to_numeric(layer_data[column], errors="coerce")
        finite_values = numeric_values.dropna()
        if finite_values.empty:
            return
        lower = float(finite_values.min())
        upper = float(finite_values.max())
        display_label = (
            f"{label} [{_format_number(lower, 2)}–{_format_number(upper, 2)}]"
        )
        if lower == upper:
            upper = lower + 1

        def style_function(feature):
            value = feature["properties"].get(column)
            try:
                ratio = (float(value) - lower) / (upper - lower)
                ratio = min(1.0, max(0.0, ratio))
            except (TypeError, ValueError):
                ratio = None
            if ratio is None:
                fill_color = missing_color
            else:
                start = (239, 246, 255)
                end = (29, 78, 216)
                rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))
                fill_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            return {
                "fillColor": fill_color,
                "color": "#374151",
                "weight": 0.6,
                "fillOpacity": 0.72,
            }

    tooltip_fields = list(
        dict.fromkeys(
            field
            for field in [
                "block_id",
                "land_use_label",
                "building_population_estimate",
                "population_density_per_ha",
                "fsi",
                "gsi",
                "mxi",
                "area_accessibility",
                "forest_zone_share",
                "provision_strong",
                "kindergarten_provision_strong",
                "territory_class",
                "spacematrix_morphotype_ru",
                "strelka_morphotype_ru",
                column,
            ]
            if field in layer_data.columns
        )
    )
    folium.GeoJson(
        layer_data,
        name=display_label if not categorical else label,
        show=show,
        style_function=style_function,
        highlight_function=lambda _: {"weight": 2, "color": "#111827"},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[COLUMN_LABELS.get(field, field) for field in tooltip_fields],
            sticky=False,
            labels=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=tooltip_fields,
            aliases=[COLUMN_LABELS.get(field, field) for field in tooltip_fields],
            localize=True,
        ),
    ).add_to(map_object)


def create_dashboard():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Сохранённый GeoJSON не найден: {DATA_FILE}. "
            "Сначала выполните расчётную ячейку ноутбука и сохраните pipeline results."
        )

    gdf = gpd.read_file(DATA_FILE)
    if gdf.empty:
        raise ValueError(f"В файле нет кварталов: {DATA_FILE}")
    if gdf.crs is None:
        raise ValueError(f"У GeoJSON не задана CRS: {DATA_FILE}")
    if "block_id" not in gdf.columns:
        gdf["block_id"] = gdf.index.astype(str)
    if "land_use" in gdf.columns:
        gdf["land_use_label"] = gdf["land_use"].map(_land_use_label)
    else:
        gdf["land_use_label"] = "Не определено"

    if "site_area" in gdf.columns:
        gdf["site_area_ha"] = pd.to_numeric(gdf["site_area"], errors="coerce") / 10_000
    if {"building_population_estimate", "site_area"}.issubset(gdf.columns):
        site_area_ha = pd.to_numeric(gdf["site_area"], errors="coerce") / 10_000
        population = pd.to_numeric(gdf["building_population_estimate"], errors="coerce")
        gdf["population_density_per_ha"] = population.div(site_area_ha.where(site_area_ha.gt(0)))

    map_gdf = gdf.to_crs(epsg=4326)
    bounds = map_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    map_object = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    map_layers = [
        ("land_use_label", "Землепользование", True),
        ("development_status", "Застроенность", True),
        ("territory_class", "Класс территории, включая лесной фонд", True),
        ("spacematrix_morphotype_ru", "Морфотип Spacematrix", True),
        ("strelka_morphotype_ru", "Морфотип Стрелки", True),
        ("building_population_estimate", "Расчётное население, чел.", False),
        ("population_density_per_ha", "Расчётная плотность населения, чел./га", False),
        ("fsi", "FSI", False),
        ("gsi", "GSI", False),
        ("mxi", "MXI", False),
        ("l", "L, этажность", False),
        ("osr", "OSR", False),
        ("area_accessibility", "Пешеходная доступность, мин.", False),
        ("forest_zone_share", "Доля лесного полигона в квартале", False),
        ("provision_strong", "Школьная обеспеченность", False),
        ("kindergarten_provision_strong", "Обеспеченность детсадами", False),
    ]
    for column, label, categorical in map_layers:
        _add_map_layer(
            map_object,
            map_gdf,
            column,
            label,
            categorical,
            show=column == "land_use_label",
        )
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    population = pd.to_numeric(
        gdf.get("building_population_estimate", pd.Series(dtype=float)), errors="coerce"
    )
    fsi = pd.to_numeric(gdf.get("fsi", pd.Series(dtype=float)), errors="coerce")
    total_area_ha = (
        float(gdf["site_area_ha"].sum(skipna=True))
        if "site_area_ha" in gdf.columns
        else float("nan")
    )
    total_population = float(population.sum(skipna=True)) if not population.empty else float("nan")
    median_fsi = float(fsi.median(skipna=True)) if not fsi.empty else float("nan")
    unbuilt_count = (
        int(gdf["development_status"].astype(str).str.startswith("Незастроенный").sum())
        if "development_status" in gdf.columns
        else None
    )
    forest_blocks = (
        int(pd.to_numeric(gdf["forest_zone_share"], errors="coerce").fillna(0).gt(0).sum())
        if "forest_zone_share" in gdf.columns
        else None
    )

    chart_html = []
    plotlyjs_included = False

    if "site_area_ha" in gdf.columns:
        land_use_area = (
            gdf.groupby("land_use_label", dropna=False)["site_area_ha"]
            .sum()
            .sort_values(ascending=True)
            .reset_index()
        )
        figure = px.bar(
            land_use_area,
            x="site_area_ha",
            y="land_use_label",
            orientation="h",
            title="Площадь по преобладающей функции",
            labels={"site_area_ha": "Площадь, га", "land_use_label": "Землепользование"},
        )
        chart_html.append(_plot_html(figure, include_plotlyjs=True))
        plotlyjs_included = True

    for column, title in [
        ("spacematrix_morphotype_ru", "Кварталы по морфотипу Spacematrix"),
        ("strelka_morphotype_ru", "Кварталы по морфотипу Стрелки"),
    ]:
        if column in gdf.columns:
            morphology_counts = (
                gdf[column].fillna("Нет данных").value_counts().rename_axis("morphotype").reset_index(name="blocks")
            )
            figure = px.bar(
                morphology_counts.sort_values("blocks"),
                x="blocks",
                y="morphotype",
                orientation="h",
                title=title,
                labels={"blocks": "Число кварталов", "morphotype": "Морфотип"},
            )
            chart_html.append(
                _plot_html(figure, include_plotlyjs=not plotlyjs_included)
            )
            plotlyjs_included = True

    if {"fsi", "population_density_per_ha"}.issubset(gdf.columns):
        figure = px.scatter(
            gdf,
            x="fsi",
            y="population_density_per_ha",
            hover_name="block_id",
            color="land_use_label",
            title="FSI и расчётная плотность населения",
            labels={
                "fsi": "FSI, м²/м²",
                "population_density_per_ha": "Население, чел./га",
                "land_use_label": "Землепользование",
            },
        )
        chart_html.append(_plot_html(figure, include_plotlyjs=not plotlyjs_included))
        plotlyjs_included = True

    if "area_accessibility" in gdf.columns:
        figure = px.histogram(
            gdf,
            x="area_accessibility",
            nbins=25,
            title="Распределение пешеходной доступности",
            labels={"area_accessibility": "Взвешенное время пути, мин."},
        )
        chart_html.append(_plot_html(figure, include_plotlyjs=not plotlyjs_included))
        plotlyjs_included = True

    if "forest_zone_share" in gdf.columns:
        figure = px.histogram(
            gdf,
            x="forest_zone_share",
            nbins=20,
            title="Доля квартала, пересекающаяся с лесным фондом",
            labels={"forest_zone_share": "Доля площади квартала"},
        )
        chart_html.append(_plot_html(figure, include_plotlyjs=not plotlyjs_included))
        plotlyjs_included = True

    if not chart_html:
        chart_html.append('<p class="empty-state">В GeoJSON недостаточно показателей для графиков.</p>')

    morphology_column = (
        "strelka_morphotype_ru"
        if "strelka_morphotype_ru" in gdf.columns
        else "spacematrix_morphotype_ru"
    )
    table_columns = [
        "block_id",
        "land_use_label",
        morphology_column,
        "spacematrix_morphotype_ru" if morphology_column != "spacematrix_morphotype_ru" else "strelka_morphotype_ru",
        "development_status",
        "territory_class",
        "site_area_ha",
        "building_population_estimate",
        "population_density_per_ha",
    ] + [
        column
        for column in gdf.columns
        if column not in {
            "geometry",
            "block_id",
            "land_use",
            "land_use_label",
            "spacematrix_morphotype_ru",
            "strelka_morphotype_ru",
            "development_status",
            "territory_class",
            "site_area_ha",
            "population_density_per_ha",
        }
    ]
    table_html = _table_html(gdf, table_columns, "block-table")
    csv_payload = base64.b64encode(
        gdf.drop(columns="geometry", errors="ignore")
        .to_csv(index=False)
        .encode("utf-8-sig")
    ).decode("ascii")
    geojson_download = DATA_FILE.relative_to(BASE_DIR).as_posix()
    land_use_options = sorted(gdf["land_use_label"].dropna().astype(str).unique())
    morphology_options = (
        sorted(gdf[morphology_column].dropna().astype(str).unique())
        if morphology_column in gdf.columns
        else []
    )
    select_html = lambda identifier, options, label: (
        f'<label>{label}<select id="{identifier}"><option value="">Все</option>'
        + "".join(
            f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
            for value in options
        )
        + "</select></label>"
    )

    target_file = DATA_DIR / "target_education_network.csv"
    target_data = pd.DataFrame()
    if target_file.exists():
        target_data = pd.read_csv(target_file)
        if "matches_analysis_boundary" in target_data.columns:
            matching_boundary = target_data["matches_analysis_boundary"].astype(str).str.lower().eq("true")
            target_data = target_data.loc[matching_boundary].copy()
        target_columns = [
            "population_scenario",
            "population",
            "population_reference_date",
            "territory_scope",
            "population_source",
            "matches_analysis_boundary",
            "service_type",
            "norm_places_per_1000",
            "target_places",
            "equivalent_facilities_at_minimum_capacity",
            "observed_capacity",
            "modelled_capacity",
            "remaining_target_places",
        ]
        target_html = _table_html(target_data, target_columns, "target-table")
    else:
        target_html = (
            '<p class="empty-state">Таблица целевых мест пока отсутствует. '
            "Сохраните её из расчётной ячейки ноутбука.</p>"
        )

    comparison_data = (
        target_data.loc[
            target_data["population_scenario"].isin(
                [
                    "МО Советское городское поселение (оценка на начало 2025)",
                    "Оценка BlocksNet по зданиям",
                ]
            )
        ]
        if "population_scenario" in target_data.columns
        else pd.DataFrame()
    )
    comparison_rows = []
    scenario_names = {
        "official": "МО Советское городское поселение (оценка на начало 2025)",
        "model": "Оценка BlocksNet по зданиям",
    }
    official_scenario_rows = comparison_data.loc[
        comparison_data["population_scenario"].eq(scenario_names["official"])
    ] if "population_scenario" in comparison_data.columns else pd.DataFrame()
    model_scenario_rows = comparison_data.loc[
        comparison_data["population_scenario"].eq(scenario_names["model"])
    ] if "population_scenario" in comparison_data.columns else pd.DataFrame()
    official_population_label = (
        _format_number(official_scenario_rows["population"].iloc[0], 0)
        if not official_scenario_rows.empty
        else "—"
    )
    model_population_label = (
        _format_number(model_scenario_rows["population"].iloc[0], 0)
        if not model_scenario_rows.empty
        else "—"
    )
    service_labels = {"school": "Школы", "kindergarten": "Детские сады"}
    for service_type, service_label in service_labels.items():
        service_rows = (
            comparison_data.loc[comparison_data["service_type"].eq(service_type)]
            if "service_type" in comparison_data.columns
            else pd.DataFrame()
        )
        official_rows = service_rows.loc[
            service_rows["population_scenario"].eq(scenario_names["official"])
        ]
        model_rows = service_rows.loc[
            service_rows["population_scenario"].eq(scenario_names["model"])
        ]
        if official_rows.empty or model_rows.empty:
            continue
        official = official_rows.iloc[0]
        model = model_rows.iloc[0]
        target_places = int(official["target_places"])
        model_places = int(model["target_places"])
        difference = model_places - target_places
        comparison_rows.append(
            {
                "Тип учреждения": service_label,
                "Норматив, мест на 1 000 жителей": _format_number(
                    official["norm_places_per_1000"], 0
                ),
                f"Целевые места для МО ({official_population_label} чел.)": _format_number(
                    target_places, 0
                ),
                f"Целевые места по модели ({model_population_label} чел.)": _format_number(
                    model_places, 0
                ),
                "Разница модели и ориентира, мест": (
                    f"{difference:+,}".replace(",", " ") if difference else "0"
                ),
                "Эквивалент учреждений для всей сети": _format_number(
                    official["equivalent_facilities_at_minimum_capacity"], 0
                ),
            }
        )
    if comparison_rows:
        comparison_html = pd.DataFrame(comparison_rows).to_html(
            index=False,
            escape=True,
            classes="data-table comparison-table",
            table_id="education-comparison-table",
            border=0,
        )
    else:
        comparison_html = (
            '<p class="empty-state">Нет совпадающих сценариев населения и модели '
            "для сравнения целевой потребности.</p>"
        )

    occupancy_tables = []
    for service_type, title in [
        ("school", "Школы: вместимость и загруженность"),
        ("kindergarten", "Детские сады: вместимость и загруженность"),
    ]:
        occupancy_file = DATA_DIR / f"{service_type}_occupancy.csv"
        if occupancy_file.exists():
            occupancy_data = pd.read_csv(occupancy_file)
            occupancy_tables.append(
                f"<h3>{title}</h3>"
                + _table_html(
                    occupancy_data,
                    [
                        "name",
                        "capacity_source",
                        "capacity",
                        "enrollment",
                        "vacant_capacity",
                        "occupancy_rate",
                        "capacity_is_imputed",
                    ],
                    f"{service_type}-occupancy-table",
                )
            )
    occupancy_html = "\n".join(occupancy_tables) or (
        '<p class="empty-state">Сохранённые таблицы фактической загруженности не найдены.</p>'
    )

    kpi_cards = [
        ("Кварталы", f"{len(gdf):,}".replace(",", " ")),
        ("Площадь территории, га", _format_number(total_area_ha, 1)),
        ("Расчётное население, чел.", _format_number(total_population, 0)),
        ("Медианный FSI", _format_number(median_fsi, 3)),
        ("Незастроенные кварталы", str(unbuilt_count) if unbuilt_count is not None else "нет поля"),
        ("Кварталы с лесным пересечением", str(forest_blocks) if forest_blocks is not None else "нет поля"),
    ]
    if {"population_scenario", "population"}.issubset(target_data.columns):
        settlement_rows = target_data.loc[
            target_data["population_scenario"].eq(
                "МО Советское городское поселение (оценка на начало 2025)"
            )
        ]
        if not settlement_rows.empty:
            official_population = float(settlement_rows["population"].iloc[0])
            kpi_cards.insert(
                3,
                (
                    "Население МО (оценка на начало 2025 г.)",
                    _format_number(official_population, 0),
                ),
            )
            if pd.notna(total_population) and official_population > 0:
                difference_percent = (total_population / official_population - 1) * 100
                kpi_cards.insert(
                    4,
                    (
                        "Отклонение модели от оценки населения МО",
                        f"{_format_number(difference_percent, 1)}%",
                    ),
                )
    kpi_html = "".join(
        f'<article class="kpi"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></article>'
        for label, value in kpi_cards
    )
    charts_markup = "\n".join(f'<section class="chart">{figure}</section>' for figure in chart_html)
    map_html = map_object.get_root().render()

    template = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Советское городское поселение · BlocksNet</title>
  <style>
    :root { color-scheme: light; --ink:#17212b; --muted:#667085; --line:#dce2e8; --panel:#fff; --accent:#2563eb; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f3f6f9; color:var(--ink); font:15px/1.5 "Segoe UI",Arial,sans-serif; }
    main { max-width:1500px; margin:auto; padding:24px; }
    header { margin:0 0 20px; }
    h1 { margin:0 0 6px; font-size:clamp(1.7rem,3vw,2.4rem); }
    h2 { margin:0 0 16px; font-size:1.35rem; }
    h3 { margin:18px 0 10px; font-size:1.05rem; }
    .muted,.note { color:var(--muted); }
    .note { background:#eaf1ff; padding:12px 16px; border-left:4px solid var(--accent); border-radius:6px; }
    .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }
    .kpi,.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 6px #17212b0a; }
    .kpi { padding:16px; display:flex; flex-direction:column; gap:8px; }
    .kpi span { color:var(--muted); font-size:.9rem; }
    .kpi strong { font-size:1.45rem; }
    .panel { padding:20px; margin:16px 0; }
    .map-frame { height:620px; border:0; width:100%; }
    .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr)); gap:12px; }
    .chart { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; min-width:0; overflow:hidden; }
    .table-controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:12px 0; }
    label { display:grid; gap:4px; color:var(--muted); font-size:.9rem; }
    input,select { min-width:210px; padding:9px 10px; border:1px solid var(--line); border-radius:6px; background:white; color:var(--ink); }
    .table-wrap { overflow:auto; max-height:650px; border:1px solid var(--line); border-radius:8px; }
    .data-table { width:100%; border-collapse:collapse; white-space:nowrap; font-size:.88rem; }
    .data-table th,.data-table td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); }
    .data-table th { position:sticky; top:0; background:#edf2f7; z-index:1; }
    .data-table tr:hover td { background:#f5f8fc; }
    .education-comparison { border:2px solid #93c5fd; background:#eff6ff; }
    .education-comparison .note { margin:0 0 12px; }
    .comparison-table th { white-space:normal; min-width:150px; }
    .comparison-table td { font-weight:600; }
    .empty-state { color:var(--muted); padding:12px; }
    .download-link { display:inline-block; padding:9px 12px; border:1px solid var(--line); border-radius:6px; background:white; color:var(--accent); text-decoration:none; }
    .download-link:hover { background:#edf2f7; }
    footer { color:var(--muted); font-size:.9rem; padding:8px 0 24px; }
    @media(max-width:700px) { main { padding:14px; } .panel { padding:14px; } .map-frame { height:520px; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Советское городское поселение · квартальная модель</h1>
    <div class="muted">BlocksNet · показатели кварталов и территории</div>
  </header>
  <p class="note">Расчётное население — оценка по данным зданий, не перепись. Загруженность услугами рассчитана по сохранённым полям: перед интерпретацией как фактической проверьте источники и годы вместимости и числа учащихся/воспитанников. Земли лесного фонда показаны по отдельному полигональному слою и не трактуются как застроенная территория. Модельная цена земли не рассчитывается: для неё нет входных данных.</p>
  <section class="kpis">__KPI_CARDS__</section>
  <section class="panel">
    <h2>Планировочный ориентир: места в школах и детских садах</h2>
    <div class="education-comparison">
      <p class="note">Сравнение всей территории МО по расчётным нормативам BlocksNet из ноутбука: потребность при оценке населения МО на начало 2025 года и при модельном населении кварталов. Это общее количество мест по нормативу, а не дефицит мест и не готовое число новых зданий для строительства; нормативы не следует считать подтверждёнными региональными нормативами проектирования.</p>
      <div class="table-wrap">__EDUCATION_COMPARISON__</div>
      <p class="muted">«Эквивалент учреждений» — ориентир по минимальной вместимости конфигурации BlocksNet (школа — 250 мест, детский сад — 80 мест) для всей сети МО, не число новых объектов. Перед решением о строительстве необходимо подтвердить актуальную вместимость действующих учреждений, фактическое число учащихся/воспитанников и доступность по населённым пунктам. Разницу между сценариями нельзя напрямую трактовать как потребность в новой школе или детском комбинате.</p>
    </div>
  </section>
  <section class="panel">
    <h2>Карта кварталов</h2>
    <p class="muted">Выберите тематический слой в списке справа; чтобы не перекрывать данные, оставляйте включённым один тематический слой. При наведении доступны атрибуты квартала.</p>
    <iframe class="map-frame" title="Интерактивная карта кварталов" srcdoc="__MAP_HTML__"></iframe>
  </section>
  <section class="panel">
    <h2>Структура и показатели</h2>
    <div class="charts">__CHARTS__</div>
  </section>
  <section class="panel">
    <h2>Целевые места в школах и детских садах</h2>
    <p class="muted">Для границы текущего анализа показаны сценарии с совпадающим охватом территории. Вместимость без автоподстановки отделена от модельной; источник данных требует отдельной проверки.</p>
    <div class="table-wrap">__TARGET_TABLE__</div>
  </section>
  <section class="panel">
    <h2>Загруженность учреждений по сохранённым данным</h2>
    <p class="muted">Загруженность рассчитана по сохранённым полям capacity и enrollment. Источник и модельный статус вместимости приведены в таблицах.</p>
    <div class="table-wrap">__OCCUPANCY_TABLES__</div>
  </section>
  <section class="panel">
    <h2>Кварталы в выборке</h2>
    <div class="table-controls">
      <label>Поиск по таблице<input id="table-search" type="search" placeholder="ID, землепользование, показатель…"></label>
      __LAND_USE_SELECT__
      __MORPHOLOGY_SELECT__
      <a class="download-link" download="sovetsky_pipeline_results.csv" href="data:text/csv;charset=utf-8;base64,__CSV_PAYLOAD__">Скачать CSV</a>
      <a class="download-link" download href="__GEOJSON_DOWNLOAD__">Скачать GeoJSON</a>
      <span class="muted" id="row-count"></span>
    </div>
    <div class="table-wrap">__BLOCK_TABLE__</div>
  </section>
  <footer>
    Источник данных — сохранённый sovetsky_pipeline_results.geojson и вспомогательные CSV из data/sovetsky.
    Подложка карты CartoDB и библиотеки Folium/Leaflet требуют доступа к интернету; диаграммы Plotly встроены в HTML.
  </footer>
</main>
<script>
  const search = document.getElementById("table-search");
  const landUse = document.getElementById("land-use-filter");
  const morphology = document.getElementById("morphology-filter");
  const rows = [...document.querySelectorAll("#block-table tbody tr")];
  const count = document.getElementById("row-count");
  function filterRows() {
    const query = search.value.trim().toLocaleLowerCase("ru");
    const land = landUse.value;
    const morph = morphology.value;
    let visible = 0;
    for (const row of rows) {
      const cells = row.cells;
      const matches = (!query || row.textContent.toLocaleLowerCase("ru").includes(query))
        && (!land || cells[1].textContent === land)
        && (!morph || cells[2].textContent === morph);
      row.hidden = !matches;
      if (matches) visible++;
    }
    count.textContent = `Показано ${visible} из ${rows.length} кварталов`;
  }
  search.addEventListener("input", filterRows);
  landUse.addEventListener("change", filterRows);
  morphology.addEventListener("change", filterRows);
  filterRows();
</script>
</body>
</html>"""

    replacements = {
        "__KPI_CARDS__": kpi_html,
        "__MAP_HTML__": html.escape(map_html, quote=True),
        "__CHARTS__": charts_markup,
        "__EDUCATION_COMPARISON__": comparison_html,
        "__TARGET_TABLE__": target_html,
        "__OCCUPANCY_TABLES__": occupancy_html,
        "__LAND_USE_SELECT__": select_html("land-use-filter", land_use_options, "Землепользование"),
        "__MORPHOLOGY_SELECT__": select_html("morphology-filter", morphology_options, "Морфотип Стрелки"),
        "__CSV_PAYLOAD__": csv_payload,
        "__GEOJSON_DOWNLOAD__": html.escape(geojson_download, quote=True),
        "__BLOCK_TABLE__": table_html,
    }
    dashboard_html = template
    for placeholder, value in replacements.items():
        dashboard_html = dashboard_html.replace(placeholder, value)

    OUTPUT_FILE.write_text(dashboard_html, encoding="utf-8")
    print(f"Дашборд сохранён: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_dashboard()

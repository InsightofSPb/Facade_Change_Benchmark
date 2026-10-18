#!/usr/bin/env python3
"""Build the local facade atlas from an existing collector bundle and serve it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

DATE = "2026-10-18"
MAX_MEMBERS = 10000
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
TEMPLATES = (
    "index.html", f"{DATE}-app.js", f"{DATE}-logic.js",
    f"{DATE}-reporter.js", f"{DATE}-style.css",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight_dependencies():
    require(sys.version_info >= (3, 9), "Нужен Python 3.9 или новее.")
    missing = []
    for name, label in (("numpy", "numpy"), ("scipy", "scipy"), ("PIL.Image", "Pillow")):
        try:
            module = importlib.import_module(name)
            if name == "PIL.Image" and not all(hasattr(module, flag) for flag in ("Resampling", "Transform")):
                missing.append("Pillow >= 9.1")
        except ImportError:
            missing.append(label)
    require(not missing, "В выбранном Python недоступны: " + ", ".join(missing) +
            ". Запустите скрипт в существующей среде с этими пакетами; скрипт не устанавливает зависимости.")
    features = importlib.import_module("PIL.features")
    require(features.check("webp"), "В установленной сборке Pillow нет поддержки WebP; выберите среду с поддержкой WebP.")


def extract_bundle(archive, destination):
    """Unpack regular files only, with bounded size and canonical relative paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                name = member.name
                parts = name.split("/")
                require(not name.startswith("/") and "\\" not in name and
                        all(part not in ("", ".", "..") and ":" not in part for part in parts),
                        f"Недопустимый путь в архиве: {name!r}")
                canonical = PurePosixPath(name).as_posix()
                require(canonical not in seen, f"Повторный путь в архиве: {name}")
                require(member.isfile(), f"В архиве разрешены только обычные файлы: {name}")
                require(len(seen) < MAX_MEMBERS, f"Архив содержит более {MAX_MEMBERS} файлов.")
                require(0 <= member.size <= MAX_MEMBER_BYTES, f"Слишком большой файл в архиве: {name}")
                total += member.size
                require(total <= MAX_TOTAL_BYTES, "Распакованный архив превышает 4 GiB.")
                seen.add(canonical)
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                require(source is not None, f"Не удалось прочитать файл архива: {name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                require(target.stat().st_size == member.size, f"Обрезанный файл архива: {name}")
    except tarfile.TarError as error:
        raise ValueError(f"Не удалось прочитать tar.gz: {error}") from error
    return {"files": len(seen), "uncompressed_bytes": total}


def check_input_contract(source):
    for relative in ("manifest.json", f"{DATE}-bundle.json", f"{DATE}-originals.json", "source/audit.json"):
        require((source / relative).is_file(),
                f"В архиве отсутствует {relative}. Соберите его collector-скриптом с --include-originals.")
    manifest = read_json(source / "manifest.json")
    require(manifest.get("kind") == "real_temporal_prepared" and manifest.get("schema_version") == 1,
            "Ожидается manifest.json формата real_temporal_prepared, schema_version=1.")
    sequences = manifest.get("sequences", [])
    require(bool(sequences), "В подготовленном манифесте нет историй фасадов.")
    for sequence in sequences:
        require(sequence.get("shape") == [512, 512] and sequence.get("crop_xywh", [])[2:] == [512, 512],
                "Этот интерфейс поддерживает подготовленные кропы 512×512.")


def prepare_site(archive, output, repository=None):
    repository = Path(repository) if repository else Path(__file__).resolve().parents[1]
    archive, output = Path(archive).resolve(), Path(os.path.abspath(output))
    require(archive.is_file(), f"Архив не найден: {archive}")
    require(not os.path.lexists(output), f"Каталог результата уже существует: {output}. Выберите новый --out или используйте --serve.")
    preflight_dependencies()
    exporter = repository / "scripts" / f"{DATE}-facade-atlas-export.py"
    template_root = repository / "web/facade_atlas"
    locations = repository / "configs" / f"{DATE}-facade-atlas-locations.json"
    sources = [exporter, Path(__file__).resolve(), locations, *(template_root / name for name in TEMPLATES)]
    for source in sources:
        require(source.is_file(), f"Не найден исходник прототипа: {source}")
    archive_digest = file_sha256(archive)
    source_hashes = {path.relative_to(repository).as_posix(): file_sha256(path) for path in sources
                     if path.is_relative_to(repository)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Exclusive reservation, including concurrent launches.
    record = {
        "schema_version": 1, "kind": "facade_atlas_local_run", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_archive_sha256": archive_digest, "source_sha256": source_hashes,
        "configuration": {"crop_size": [512, 512], "minimum_component_pixels": 64,
                          "model_inference": False, "alignment_estimation": False},
    }
    def save_record():
        (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_record()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-input-", dir=output.parent) as temporary:
            unpacked = Path(temporary) / "input"
            record["input_archive"] = extract_bundle(archive, unpacked)
            require(file_sha256(archive) == archive_digest, "Архив изменился во время чтения; повторите запуск с неизменным файлом.")
            check_input_contract(unpacked)
            command = [sys.executable, str(exporter), "--input", str(unpacked), "--dist", str(output)]
            result = subprocess.run(command, check=False)
            require(result.returncode == 0,
                    f"Экспорт данных завершился с кодом {result.returncode}; подробности выше. Для повтора выберите новый --out.")
            for name in TEMPLATES:
                shutil.copyfile(template_root / name, output / name)
            shutil.copyfile(locations, output / "data" / f"{DATE}-locations.json")
            validation = read_json(output / "data" / f"{DATE}-validation.json")
            require(validation.get("status") == "passed", "Экспортёр не подтвердил проверку данных.")
            record["summary"] = {key: validation[key] for key in ("facades", "frames", "adjacent_pairs", "numeric_frames_checked")}
            record["status"] = "completed"
            save_record()
    except BaseException as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        save_record()
        raise
    print(f"Прототип подготовлен: {output}", flush=True)
    return output


class AtlasHandler(SimpleHTTPRequestHandler):
    """Expose only generated browser resources, with no directory listing."""
    def send_head(self):
        root = Path(self.directory).resolve()
        target = Path(super().translate_path(self.path)).resolve()
        if not target.is_relative_to(root):
            self.send_error(404)
            return None
        relative = target.relative_to(root)
        allowed = not relative.parts or relative.as_posix() in TEMPLATES or relative.parts[0] in ("assets", "data")
        if not allowed:
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404, "Directory listing is disabled")
        return None


def make_server(directory, port):
    directory = Path(directory).resolve()
    require(directory.is_dir(), f"Каталог прототипа не найден: {directory}")
    require((directory / "index.html").is_file() and (directory / "data" / f"{DATE}-facades.json").is_file(),
            "В каталоге нет подготовленного прототипа (index.html и data/2026-10-18-facades.json).")
    require((directory / "run.json").is_file(), "Нет run.json; сначала подготовьте прототип через --bundle и --out.")
    record = read_json(directory / "run.json")
    require(record.get("kind") == "facade_atlas_local_run" and record.get("status") == "completed",
            "run.json не подтверждает завершённую подготовку прототипа.")
    return ThreadingHTTPServer(("127.0.0.1", port), partial(AtlasHandler, directory=str(directory)))


def serve_site(directory, port):
    with make_server(directory, port) as server:
        print(f"Откройте http://127.0.0.1:{server.server_port}/ . Для остановки нажмите Ctrl+C.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nЛокальный сервер остановлен.", flush=True)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Подготовить и открыть локальный прототип истории фасадов без инференса.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path, help="Архив collector tar.gz, собранный с --include-originals")
    mode.add_argument("--serve", type=Path, help="Уже подготовленный каталог; экспорт повторно не запускается")
    parser.add_argument("--out", type=Path, help="Новый каталог результата")
    parser.add_argument("--prepare-only", action="store_true", help="Подготовить файлы и завершиться без HTTP-сервера")
    parser.add_argument("--port", type=int, default=8765, help="Локальный порт (по умолчанию 8765)")
    arguments = parser.parse_args(arguments)
    if arguments.bundle and not arguments.out:
        parser.error("--bundle требует --out")
    if arguments.serve and (arguments.out or arguments.prepare_only):
        parser.error("--serve нельзя объединять с --out или --prepare-only")
    if not 1 <= arguments.port <= 65535:
        parser.error("--port должен быть от 1 до 65535")
    return arguments


def main(arguments=None):
    options = parse_args(arguments)
    try:
        if options.serve:
            serve_site(options.serve, options.port)
        else:
            output = prepare_site(options.bundle, options.out)
            if not options.prepare_only:
                serve_site(output, options.port)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

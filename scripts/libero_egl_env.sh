#!/usr/bin/env bash
# MuJoCo EGL headless rendering in conda can crash at import time with:
#   Unable to find symbol malloc. Aborting.
# Preloading the system libc lets PyOpenGL resolve malloc when loading EGL/OpenGL.

if [[ -z "${LIBERO_EGL_LD_PRELOAD:-}" ]]; then
  for _libc in /usr/lib/x86_64-linux-gnu/libc.so.6 /lib/x86_64-linux-gnu/libc.so.6; do
    if [[ -f "${_libc}" ]]; then
      export LIBERO_EGL_LD_PRELOAD="${_libc}"
      break
    fi
  done
fi

if [[ -n "${LIBERO_EGL_LD_PRELOAD:-}" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${LIBERO_EGL_LD_PRELOAD}:"*) ;;
    *)
      if [[ -n "${LD_PRELOAD:-}" ]]; then
        export LD_PRELOAD="${LIBERO_EGL_LD_PRELOAD}:${LD_PRELOAD}"
      else
        export LD_PRELOAD="${LIBERO_EGL_LD_PRELOAD}"
      fi
      ;;
  esac
fi

"""Modular high-performance 3D spatial web application engine.

Integrates Three.js, GSAP, and sleek dark-mode glassmorphic styling into Streamlit.
Provides state-reactive 3D scenes (auth portal centerpiece & persistent dashboard depth canvas),
interactive 3D card tilt physics, specular cursor sheen, and 60fps lifecycle management.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def get_spatial_css(*, authenticated: bool = True) -> str:
    """Return high-end spatial dark-mode glassmorphic CSS rules."""
    hide_sidebar_css = (
        """
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        """
        if not authenticated
        else ""
    )

    return f"""
    <style>
        {hide_sidebar_css}

        :root {{
            --spatial-bg: #06060A;
            --spatial-surface: rgba(10, 15, 29, 0.55);
            --spatial-surface-hover: rgba(17, 24, 39, 0.72);
            --spatial-border: rgba(255, 255, 255, 0.12);
            --spatial-border-glow: rgba(56, 189, 248, 0.38);
            --spatial-cyan: #00F2FE;
            --spatial-violet: #8B5CF6;
            --spatial-crimson: #FF3B30;
            --specular-x: 50%;
            --specular-y: 50%;
        }}

        /* Deep Obsidian Space Background with multi-layer ambient glows */
        [data-testid="stAppViewContainer"] {{
            background-color: var(--spatial-bg) !important;
            background-image:
                radial-gradient(circle at 18% 14%, rgba(56, 189, 248, 0.12), transparent 42%),
                radial-gradient(circle at 82% 16%, rgba(139, 92, 246, 0.14), transparent 40%),
                radial-gradient(circle at 50% 88%, rgba(16, 185, 129, 0.08), transparent 45%);
            background-attachment: fixed;
            background-size: cover;
            color: #F8FAFC;
        }}

        /* Noise overlay layer for tactile luxury depth */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: radial-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 0);
            background-size: 24px 24px;
            pointer-events: none;
            z-index: 1;
            opacity: 0.65;
        }}

        .block-container {{
            position: relative;
            z-index: 2;
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
        }}

        /* Header transparency */
        [data-testid="stHeader"] {{
            background: rgba(6, 6, 10, 0.45) !important;
            backdrop-filter: blur(16px) !important;
            border-bottom: 1px solid var(--spatial-border);
        }}

        /* Sidebar Obsidian Glassmorphism */
        [data-testid="stSidebar"] {{
            background: rgba(8, 12, 22, 0.72) !important;
            backdrop-filter: blur(24px) saturate(190%) !important;
            border-right: 1px solid var(--spatial-border);
            box-shadow: 10px 0 35px -10px rgba(0, 0, 0, 0.85);
        }}

        /* 3D Glassmorphic Auth Portal Card */
        .auth-portal-card-3d {{
            position: relative;
            border-radius: 20px;
            padding: 2.2rem 2.4rem;
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.58), rgba(6, 10, 20, 0.72));
            backdrop-filter: blur(28px) saturate(200%);
            border: 1px solid var(--spatial-border);
            box-shadow:
                0 30px 60px -15px rgba(0, 0, 0, 0.85),
                inset 0 1px 0 rgba(255, 255, 255, 0.18),
                0 0 45px rgba(56, 189, 248, 0.12);
            transform-style: preserve-3d;
            perspective: 1200px;
            transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1),
                        box-shadow 0.3s ease,
                        border-color 0.3s ease;
        }}

        .auth-portal-card-3d:hover {{
            border-color: rgba(56, 189, 248, 0.45);
            box-shadow:
                0 40px 75px -18px rgba(0, 0, 0, 0.95),
                inset 0 1px 0 rgba(255, 255, 255, 0.3),
                0 0 60px rgba(56, 189, 248, 0.22);
        }}

        /* Hero Header 3D Spatial Banner */
        .hero-banner-3d {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
            padding: 1.15rem 1.45rem;
            margin-bottom: 1.25rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.65), rgba(8, 12, 24, 0.52));
            backdrop-filter: blur(24px) saturate(190%);
            border: 1px solid var(--spatial-border);
            box-shadow:
                0 24px 48px -20px rgba(0, 0, 0, 0.85),
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 0 40px rgba(56, 189, 248, 0.1);
            transform-style: preserve-3d;
            transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.3s ease;
        }}
        .hero-banner-3d:hover {{
            transform: perspective(1000px) translateY(-3px) scale3d(1.008, 1.008, 1.008);
            border-color: rgba(56, 189, 248, 0.42);
        }}
        .hero-banner-title {{
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0;
            background: linear-gradient(90deg, #F8FAFC, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .finance-subtitle {{
            color: #94A3B8;
            font-size: 0.92rem;
            margin-top: 0.3rem;
        }}

        /* 3D Glassmorphic KPI Cards with Specular Highlight */
        [data-testid="stMetric"] {{
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            background: linear-gradient(150deg, rgba(15, 23, 42, 0.6), rgba(6, 10, 20, 0.5));
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--spatial-border);
            box-shadow:
                0 20px 42px -16px rgba(0, 0, 0, 0.8),
                inset 0 1px 0 rgba(255, 255, 255, 0.18);
            transform-style: preserve-3d;
            perspective: 1000px;
            transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1),
                        box-shadow 0.3s ease,
                        border-color 0.3s ease;
        }}

        /* Specular light glare follows mouse coordinate CSS vars */
        [data-testid="stMetric"]::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(
                circle 160px at var(--specular-x, 50%) var(--specular-y, 50%),
                rgba(255, 255, 255, 0.14),
                transparent 70%
            );
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.25s ease;
        }}
        [data-testid="stMetric"]:hover::after {{
            opacity: 1;
        }}

        [data-testid="stMetric"]:hover {{
            transform: perspective(1000px) translateY(-6px) rotateX(4deg) rotateY(-2deg) scale(1.02);
            border-color: rgba(56, 189, 248, 0.6);
            box-shadow:
                0 32px 55px -16px rgba(0, 0, 0, 0.95),
                0 0 35px rgba(56, 189, 248, 0.28),
                inset 0 1px 0 rgba(255, 255, 255, 0.32);
        }}

        /* 3D Depth for Expanders, Plotly Charts, Forms */
        [data-testid="stExpander"] {{
            border-radius: 16px !important;
            border: 1px solid var(--spatial-border) !important;
            background: rgba(10, 15, 29, 0.5) !important;
            backdrop-filter: blur(18px) saturate(180%) !important;
            box-shadow: 0 16px 36px -18px rgba(0, 0, 0, 0.75);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }}
        [data-testid="stExpander"]:hover {{
            border-color: rgba(56, 189, 248, 0.35) !important;
            box-shadow: 0 24px 48px -18px rgba(0, 0, 0, 0.88), 0 0 25px rgba(56, 189, 248, 0.12);
        }}

        [data-testid="stPlotlyChart"] {{
            border-radius: 18px;
            padding: 0.4rem 0.6rem;
            background: rgba(8, 12, 22, 0.45);
            backdrop-filter: blur(16px);
            border: 1px solid var(--spatial-border);
            box-shadow:
                0 20px 42px -20px rgba(0, 0, 0, 0.8),
                inset 0 1px 0 rgba(255, 255, 255, 0.12);
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        [data-testid="stPlotlyChart"]:hover {{
            border-color: rgba(56, 189, 248, 0.38);
            box-shadow: 0 28px 56px -22px rgba(0, 0, 0, 0.9), 0 0 30px rgba(56, 189, 248, 0.14);
        }}

        /* Modern Glowing Input Fields */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: rgba(15, 23, 42, 0.65) !important;
            border-radius: 12px !important;
            border: 1px solid var(--spatial-border) !important;
            backdrop-filter: blur(12px) !important;
            color: #F8FAFC !important;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease !important;
        }}
        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div {{
            border-color: #00F2FE !important;
            box-shadow: 0 0 22px rgba(0, 242, 254, 0.32), inset 0 0 10px rgba(0, 242, 254, 0.08) !important;
            transform: translateY(-1px);
        }}

        /* Primary CTA Buttons with Magnetic Spring Depth */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            box-shadow: 0 10px 24px -10px rgba(0, 0, 0, 0.8) !important;
            transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1),
                        box-shadow 0.25s ease,
                        background 0.25s ease !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-3px) scale(1.015) !important;
            box-shadow: 0 18px 34px -10px rgba(56, 189, 248, 0.45), 0 0 25px rgba(56, 189, 248, 0.25) !important;
        }}
        .stButton > button:active, .stDownloadButton > button:active {{
            transform: translateY(1px) scale(0.99) !important;
        }}

        /* Subtle Card Shake Animation for Auth Error */
        @keyframes spatialErrorShake {{
            0%, 100% {{ transform: translateX(0); }}
            15%, 45%, 75% {{ transform: translateX(-12px); }}
            30%, 60%, 90% {{ transform: translateX(12px); }}
        }}
        .spatial-shake {{
            animation: spatialErrorShake 0.45s cubic-bezier(0.36, 0.07, 0.19, 0.97) both;
            border-color: #FF3B30 !important;
            box-shadow: 0 0 35px rgba(255, 59, 48, 0.45) !important;
        }}

        /* 3D Cube Emblem */
        .auth-hero-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-top: 0.4rem;
            margin-bottom: 1.4rem;
            animation: floatIn3D 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        .cube-stage {{
            width: 72px;
            height: 72px;
            perspective: 520px;
            margin-bottom: 1rem;
        }}
        .cube-stage-sm {{
            width: 44px;
            height: 44px;
            perspective: 360px;
            flex-shrink: 0;
        }}
        .cube-3d {{
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            animation: cubeSpin3D 10s linear infinite;
        }}
        .cube-face {{
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.35rem;
            color: #F8FAFC;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.8), rgba(139, 92, 246, 0.75));
            border: 1px solid rgba(255, 255, 255, 0.45);
            border-radius: 12px;
            box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.25), 0 10px 28px rgba(0, 0, 0, 0.6);
            backface-visibility: visible;
        }}
        .cube-stage-sm .cube-face {{
            font-size: 0.9rem;
            border-radius: 8px;
        }}
        .cube-front  {{ transform: rotateY(0deg) translateZ(36px); }}
        .cube-back   {{ transform: rotateY(180deg) translateZ(36px); }}
        .cube-right  {{ transform: rotateY(90deg) translateZ(36px); }}
        .cube-left   {{ transform: rotateY(-90deg) translateZ(36px); }}
        .cube-top    {{ transform: rotateX(90deg) translateZ(36px); }}
        .cube-bottom {{ transform: rotateX(-90deg) translateZ(36px); }}

        .cube-stage-sm .cube-front  {{ transform: rotateY(0deg) translateZ(22px); }}
        .cube-stage-sm .cube-back   {{ transform: rotateY(180deg) translateZ(22px); }}
        .cube-stage-sm .cube-right  {{ transform: rotateY(90deg) translateZ(22px); }}
        .cube-stage-sm .cube-left   {{ transform: rotateY(-90deg) translateZ(22px); }}
        .cube-stage-sm .cube-top    {{ transform: rotateX(90deg) translateZ(22px); }}
        .cube-stage-sm .cube-bottom {{ transform: rotateX(-90deg) translateZ(22px); }}

        @keyframes floatIn3D {{
            0% {{
                opacity: 0;
                transform: perspective(1000px) translateY(22px) rotateX(8deg) scale(0.97);
            }}
            100% {{
                opacity: 1;
                transform: perspective(1000px) translateY(0) rotateX(0deg) scale(1);
            }}
        }}

        @keyframes cubeSpin3D {{
            0%   {{ transform: rotateX(-18deg) rotateY(0deg); }}
            50%  {{ transform: rotateX(18deg) rotateY(180deg); }}
            100% {{ transform: rotateX(-18deg) rotateY(360deg); }}
        }}

        .auth-hero-title {{
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(90deg, #F8FAFC, #38BDF8, #818CF8, #34D399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.35rem;
        }}
        .auth-hero-sub {{
            color: rgba(148, 163, 184, 0.95);
            font-size: 1rem;
            max-width: 580px;
            margin: 0 auto;
        }}

        .user-badge-card {{
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(139, 92, 246, 0.08));
            border: 1px solid var(--spatial-border);
            margin-bottom: 0.75rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15), 0 12px 24px -10px rgba(0, 0, 0, 0.6);
        }}
        .user-badge-name {{
            font-weight: 700;
            font-size: 0.98rem;
            color: #F8FAFC;
        }}
        .user-badge-contact {{
            font-size: 0.82rem;
            color: rgba(148, 163, 184, 0.95);
            word-break: break-all;
        }}
    </style>
    """


_SPATIAL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            width: 100%;
            height: 100%;
            background: transparent;
            pointer-events: none;
        }
        #spatial-local-canvas {
            position: fixed;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }
    </style>
</head>
<body>
    <canvas id="spatial-local-canvas"></canvas>
    <script>
    (function() {
        const MODE = "__MODE__";
        const STATE = "__STATE__";

        let targetDoc = document;
        let targetWin = window;
        let isParentAccessible = false;

        try {
            if (window.parent && window.parent.document && window.parent.document.body) {
                targetDoc = window.parent.document;
                targetWin = window.parent;
                isParentAccessible = true;
            }
        } catch (e) {
            isParentAccessible = false;
        }

        const prefersReducedMotion = targetWin.matchMedia && targetWin.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const isMobile = targetWin.innerWidth < 768;

        let canvas = null;
        if (isParentAccessible) {
            canvas = targetDoc.getElementById('spatial-3d-canvas-root');
            if (!canvas) {
                canvas = targetDoc.createElement('canvas');
                canvas.id = 'spatial-3d-canvas-root';
                canvas.style.cssText = 'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0; pointer-events: none; opacity: 0; transition: opacity 0.8s ease;';
                targetDoc.body.insertBefore(canvas, targetDoc.body.firstChild);
            }
        } else {
            canvas = document.getElementById('spatial-local-canvas');
        }

        if (!canvas || typeof THREE === 'undefined') return;

        const globalKey = '__SPATIAL_3D_CONTROLLER__';
        if (targetWin[globalKey]) {
            try {
                targetWin[globalKey].setMode(MODE, STATE);
                return;
            } catch (err) {
                // fall through if needed
            }
        }

        const renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            alpha: true,
            antialias: !isMobile,
            powerPreference: 'high-performance'
        });
        renderer.setPixelRatio(Math.min(targetWin.devicePixelRatio || 1, 2));
        renderer.setSize(targetWin.innerWidth, targetWin.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x06060A, MODE === 'auth' ? 0.028 : 0.022);

        const camera = new THREE.PerspectiveCamera(45, targetWin.innerWidth / targetWin.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, MODE === 'auth' ? 14 : 18);

        const ambientLight = new THREE.AmbientLight(0x0F172A, 1.2);
        scene.add(ambientLight);

        const lightCyan = new THREE.PointLight(0x00F2FE, 3.2, 55);
        lightCyan.position.set(7, 6, 8);
        scene.add(lightCyan);

        const lightViolet = new THREE.PointLight(0x8B5CF6, 3.0, 55);
        lightViolet.position.set(-7, -6, 7);
        scene.add(lightViolet);

        const lightAccent = new THREE.PointLight(0x10B981, 1.5, 40);
        lightAccent.position.set(0, 8, -4);
        scene.add(lightAccent);

        const rootGroup = new THREE.Group();
        scene.add(rootGroup);

        let torusKnot = null;
        let coreMesh = null;
        let particleSystem = null;
        let gridHelper = null;

        const particleCount = isMobile ? 180 : (MODE === 'auth' ? 520 : 380);
        const particleGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            const r = MODE === 'auth' ? (8 + Math.random() * 18) : (10 + Math.random() * 26);
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);

            const rnd = Math.random();
            if (rnd < 0.45) {
                colors[i * 3] = 0.0; colors[i * 3 + 1] = 0.95; colors[i * 3 + 2] = 1.0;
            } else if (rnd < 0.8) {
                colors[i * 3] = 0.55; colors[i * 3 + 1] = 0.36; colors[i * 3 + 2] = 0.96;
            } else {
                colors[i * 3] = 0.06; colors[i * 3 + 1] = 0.72; colors[i * 3 + 2] = 0.51;
            }
        }
        particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const particleMat = new THREE.PointsMaterial({
            size: isMobile ? 0.12 : 0.18,
            vertexColors: true,
            transparent: true,
            opacity: 0.72,
            blending: THREE.AdditiveBlending
        });
        particleSystem = new THREE.Points(particleGeo, particleMat);
        rootGroup.add(particleSystem);

        if (MODE === 'auth') {
            const torusGeo = new THREE.TorusKnotGeometry(2.7, 0.72, 120, 24, 2, 3);
            const torusMat = new THREE.MeshStandardMaterial({
                color: 0x0A0F1D,
                roughness: 0.18,
                metalness: 0.88,
                wireframe: false
            });
            torusKnot = new THREE.Mesh(torusGeo, torusMat);
            rootGroup.add(torusKnot);

            const wireGeo = new THREE.TorusKnotGeometry(2.74, 0.74, 90, 18, 2, 3);
            const wireMat = new THREE.MeshBasicMaterial({
                color: 0x00F2FE,
                wireframe: true,
                transparent: true,
                opacity: 0.28
            });
            const wireMesh = new THREE.Mesh(wireGeo, wireMat);
            torusKnot.add(wireMesh);

            const coreGeo = new THREE.IcosahedronGeometry(1.6, 2);
            const coreMat = new THREE.MeshStandardMaterial({
                color: 0x8B5CF6,
                emissive: 0x38BDF8,
                emissiveIntensity: 0.35,
                wireframe: true,
                transparent: true,
                opacity: 0.45
            });
            coreMesh = new THREE.Mesh(coreGeo, coreMat);
            rootGroup.add(coreMesh);
        } else {
            gridHelper = new THREE.GridHelper(60, 40, 0x00F2FE, 0x1E293B);
            gridHelper.position.y = -6.5;
            gridHelper.rotation.x = Math.PI * 0.05;
            gridHelper.material.transparent = true;
            gridHelper.material.opacity = 0.22;
            rootGroup.add(gridHelper);
        }

        canvas.style.opacity = '1';

        let mouseX = 0, mouseY = 0;
        let targetRotX = 0, targetRotY = 0;
        let baseSpinSpeed = 0.006;
        let currentSpinSpeed = baseSpinSpeed;

        function onMouseMove(e) {
            if (prefersReducedMotion) return;
            const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
            const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
            mouseX = (clientX / targetWin.innerWidth) * 2 - 1;
            mouseY = -(clientY / targetWin.innerHeight) * 2 + 1;
            targetRotY = mouseX * 0.35;
            targetRotX = -mouseY * 0.28;
        }
        targetDoc.addEventListener('mousemove', onMouseMove, { passive: true });

        function updateCardInteractions() {
            const cards = targetDoc.querySelectorAll('[data-testid="stMetric"], .auth-portal-card-3d, .hero-banner-3d, [data-testid="stExpander"], [data-testid="stPlotlyChart"]');
            cards.forEach(card => {
                if (card.dataset.spatialTiltAttached) return;
                card.dataset.spatialTiltAttached = 'true';

                card.addEventListener('mousemove', (e) => {
                    if (prefersReducedMotion) return;
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const normX = (x / rect.width) - 0.5;
                    const normY = (y / rect.height) - 0.5;

                    card.style.transform = 'perspective(1000px) rotateX(' + (-normY * 9) + 'deg) rotateY(' + (normX * 9) + 'deg) scale3d(1.015, 1.015, 1.015)';
                    card.style.setProperty('--specular-x', ((x / rect.width) * 100) + '%');
                    card.style.setProperty('--specular-y', ((y / rect.height) * 100) + '%');
                }, { passive: true });

                card.addEventListener('mouseleave', () => {
                    card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
                });
            });
        }
        updateCardInteractions();
        const domObserver = new MutationObserver(() => updateCardInteractions());
        domObserver.observe(targetDoc.body, { childList: true, subtree: true });

        function attachInputListeners() {
            const inputs = targetDoc.querySelectorAll('input, textarea');
            inputs.forEach(input => {
                if (input.dataset.spatialFocusAttached) return;
                input.dataset.spatialFocusAttached = 'true';

                input.addEventListener('focus', () => {
                    currentSpinSpeed = 0.022;
                    if (typeof gsap !== 'undefined') {
                        gsap.to(lightCyan, { intensity: 4.8, duration: 0.4 });
                        gsap.to(lightViolet, { intensity: 4.5, duration: 0.4 });
                    }
                });

                input.addEventListener('blur', () => {
                    currentSpinSpeed = baseSpinSpeed;
                    if (typeof gsap !== 'undefined') {
                        gsap.to(lightCyan, { intensity: 3.2, duration: 0.6 });
                        gsap.to(lightViolet, { intensity: 3.0, duration: 0.6 });
                    }
                });
            });
        }
        attachInputListeners();
        targetDoc.addEventListener('focusin', attachInputListeners);

        function triggerAuthError() {
            if (typeof gsap === 'undefined') return;
            const authCard = targetDoc.querySelector('.auth-portal-card-3d') || targetDoc.querySelector('[data-testid="stForm"]');
            if (authCard) {
                gsap.fromTo(authCard,
                    { x: -14 },
                    { x: 14, duration: 0.05, repeat: 7, yoyo: true, ease: "power1.inOut", onComplete: () => gsap.set(authCard, { x: 0 }) }
                );
            }
            gsap.to([lightCyan.color, lightViolet.color], { r: 1.0, g: 0.23, b: 0.19, duration: 0.25 });
            gsap.to([lightCyan, lightViolet], { intensity: 5.5, duration: 0.25 });
            setTimeout(() => {
                gsap.to(lightCyan.color, { r: 0.0, g: 0.95, b: 0.99, duration: 1.2 });
                gsap.to(lightViolet.color, { r: 0.55, g: 0.36, b: 0.96, duration: 1.2 });
                gsap.to(lightCyan, { intensity: 3.2, duration: 1.2 });
                gsap.to(lightViolet, { intensity: 3.0, duration: 1.2 });
            }, 1500);
        }

        function triggerLoginSuccess() {
            if (typeof gsap === 'undefined') return;
            const authCard = targetDoc.querySelector('.auth-portal-card-3d') || targetDoc.querySelector('[data-testid="stForm"]');
            if (authCard) {
                gsap.to(authCard, { opacity: 0, scale: 1.15, filter: "blur(14px)", duration: 0.9 });
            }
            gsap.to(camera.position, { z: -10, duration: 1.2, ease: "power3.in" });
            gsap.to(camera, { fov: 90, duration: 1.2, onUpdate: () => camera.updateProjectionMatrix() });
        }

        const alertObserver = new MutationObserver(mutations => {
            for (const m of mutations) {
                for (const node of m.addedNodes) {
                    if (node.nodeType === 1 && (node.matches('[data-testid="stAlert"], .stAlert') || node.querySelector('[data-testid="stAlert"]'))) {
                        triggerAuthError();
                        return;
                    }
                }
            }
        });
        alertObserver.observe(targetDoc.body, { childList: true, subtree: true });

        let animationFrameId = null;
        let clock = new THREE.Clock();

        function animate() {
            animationFrameId = requestAnimationFrame(animate);

            if (targetDoc.hidden) return;

            const delta = clock.getDelta();

            rootGroup.rotation.y += (targetRotY - rootGroup.rotation.y) * 0.05;
            rootGroup.rotation.x += (targetRotX - rootGroup.rotation.x) * 0.05;

            if (torusKnot) {
                torusKnot.rotation.x += currentSpinSpeed * 0.75;
                torusKnot.rotation.y += currentSpinSpeed;
            }
            if (coreMesh) {
                coreMesh.rotation.y -= currentSpinSpeed * 1.2;
                coreMesh.rotation.z += currentSpinSpeed * 0.5;
            }
            if (particleSystem) {
                particleSystem.rotation.y += 0.0008;
                particleSystem.rotation.x += 0.0004;
            }

            const time = clock.getElapsedTime();
            lightCyan.position.x = 7 * Math.cos(time * 0.4);
            lightCyan.position.y = 6 * Math.sin(time * 0.3);
            lightViolet.position.x = -7 * Math.cos(time * 0.35);
            lightViolet.position.y = -6 * Math.sin(time * 0.45);

            renderer.render(scene, camera);
        }
        animate();

        function onResize() {
            camera.aspect = targetWin.innerWidth / targetWin.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(targetWin.innerWidth, targetWin.innerHeight);
        }
        targetWin.addEventListener('resize', onResize);

        function destroy() {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            targetDoc.removeEventListener('mousemove', onMouseMove);
            targetWin.removeEventListener('resize', onResize);
            targetDoc.removeEventListener('focusin', attachInputListeners);
            domObserver.disconnect();
            alertObserver.disconnect();

            particleGeo.dispose();
            particleMat.dispose();
            if (torusKnot) {
                torusKnot.geometry.dispose();
                torusKnot.material.dispose();
            }
            if (coreMesh) {
                coreMesh.geometry.dispose();
                coreMesh.material.dispose();
            }
            renderer.dispose();
            if (canvas && canvas.parentNode && !isParentAccessible) {
                canvas.parentNode.removeChild(canvas);
            }
        }

        targetWin[globalKey] = {
            setMode: function(newMode, newState) {
                if (newState === 'error') triggerAuthError();
                if (newState === 'success') triggerLoginSuccess();
            },
            destroy: destroy
        };
    })();
    </script>
</body>
</html>
"""


def render_spatial_canvas(
    mode: str = "dashboard",
    state: str = "idle",
) -> None:
    """Inject the Three.js and GSAP 3D interactive spatial engine.

    Disposes of previous geometries and textures to maintain 60fps.
    Supports cursor parallax, state-reactive lighting/warp, and card tilt.
    """
    html_content = _SPATIAL_HTML_TEMPLATE.replace("__MODE__", mode).replace("__STATE__", state)
    if hasattr(st, "html"):
        try:
            st.html(html_content, unsafe_allow_javascript=True)
            return
        except TypeError:
            pass
    components.html(html_content, height=0, width=0)

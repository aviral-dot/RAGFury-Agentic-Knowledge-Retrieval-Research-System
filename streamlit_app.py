"""
RAGFury — Premium Animated Streamlit Frontend
Keeps the original RAGFury backend/API architecture intact.
"""

import os
import time
import html
import requests
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv(
    "RAGFURY_API_URL",
    "https://ragfury.vercel.app",
).rstrip("/")


st.set_page_config(
    page_title="RAGFury",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    
    
    
    /* ======================================================
   WORKSPACE INPUT VISIBILITY FIX
   ====================================================== */

section[data-testid="stSidebar"]
div[data-testid="stTextInput"]
input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;

    background: #151821 !important;

    border: 1px solid rgba(255,255,255,.18) !important;

    border-radius: 12px !important;

    height: 46px !important;
    min-height: 46px !important;

    padding: 0 14px !important;

    font-size: 14px !important;
    font-weight: 500 !important;

    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stTextInput"]
input::placeholder {
    color: #9ca3b5 !important;
    -webkit-text-fill-color: #9ca3b5 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stTextInput"]
input:focus {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;

    background: #181b27 !important;

    border-color: #9b87f5 !important;

    box-shadow:
        0 0 0 2px rgba(155,135,245,.15),
        0 0 20px rgba(155,135,245,.10) !important;
}

    /* ======================================================
       ROOT
       ====================================================== */

    :root {
        --rf-bg: #07080c;
        --rf-bg2: #0b0d12;
        --rf-panel: rgba(255,255,255,.035);
        --rf-panel-hover: rgba(255,255,255,.055);
        --rf-border: rgba(255,255,255,.085);
        --rf-text: #f5f5f7;
        --rf-muted: #858b9b;
        --rf-accent: #9b87f5;
        --rf-cyan: #63d8e8;
        --rf-green: #42d99b;
    }

    /* ======================================================
   STREAMLIT TOP HEADER / WHITE STRIP FIX
   ====================================================== */

header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

header[data-testid="stHeader"] > div {
    background: transparent !important;
}

/* Hide the top decoration/toolbar area */
[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Remove Streamlit's top blank space */
.stAppViewContainer {
    background: transparent !important;
}

.main {
    background: transparent !important;
}


    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(124,92,255,.11),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(50,200,220,.065),
                transparent 27%
            ),
            linear-gradient(
                145deg,
                #06070a 0%,
                #090b10 50%,
                #06070a 100%
            );
    }


    .stApp::before {
        content: "";
        position: fixed;
        width: 600px;
        height: 600px;
        left: -300px;
        top: -250px;
        border-radius: 50%;
        background: radial-gradient(
            circle,
            rgba(139,92,246,.12),
            transparent 68%
        );
        filter: blur(45px);
        pointer-events: none;
        animation: ambientOne 16s ease-in-out infinite alternate;
        z-index: 0;
    }


    .stApp::after {
        content: "";
        position: fixed;
        width: 550px;
        height: 550px;
        right: -250px;
        bottom: -220px;
        border-radius: 50%;
        background: radial-gradient(
            circle,
            rgba(34,211,238,.08),
            transparent 68%
        );
        filter: blur(45px);
        pointer-events: none;
        animation: ambientTwo 19s ease-in-out infinite alternate;
        z-index: 0;
    }


    @keyframes ambientOne {
        0% {
            transform: translate(0,0) scale(1);
        }

        100% {
            transform: translate(100px,70px) scale(1.2);
        }
    }


    @keyframes ambientTwo {
        0% {
            transform: translate(0,0) scale(1);
        }

        100% {
            transform: translate(-80px,-60px) scale(1.15);
        }
    }


    /* ======================================================
       MAIN WIDTH
       ====================================================== */

    .block-container {
        max-width: 1250px;
        padding-top: 1.7rem;
        padding-bottom: 7rem;
    }


    /* ======================================================
       TYPOGRAPHY
       ====================================================== */

    h1,
    h2,
    h3 {
        letter-spacing: -0.04em !important;
    }


    p {
        line-height: 1.65;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(9,10,15,.98),
                rgba(5,6,9,.98)
            );
        border-right: 1px solid rgba(255,255,255,.065);
    }


    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ======================================================
   RAGFURY BRAND LOGO
   ====================================================== */

.rf-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 4px 14px 4px;
}

.rf-brand-icon {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 13px;

    display: flex;
    align-items: center;
    justify-content: center;

    position: relative;

    background:
        radial-gradient(
            circle at 30% 25%,
            #ffffff 0%,
            #c4b5fd 18%,
            #8b5cf6 45%,
            #211642 100%
        );

    box-shadow:
        0 0 20px rgba(139,92,246,.45),
        0 0 45px rgba(139,92,246,.18);

    animation: rfLogoFloat 4s ease-in-out infinite;
}

.rf-brand-icon span {
    color: white;
    font-size: 25px;
    line-height: 1;
    font-weight: 800;

    text-shadow:
        0 0 10px rgba(255,255,255,.9);
}

.rf-brand-text {
    display: flex;
    flex-direction: column;
}

.rf-brand-name {
    font-size: 24px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.7px;

    background:
        linear-gradient(
            100deg,
            #ffffff,
            #c4b5fd,
            #67e8f9
        );

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.rf-brand-name span {
    color: #a78bfa;
}

.rf-brand-subtitle {
    margin-top: 5px;

    font-size: 9px;
    font-weight: 700;
    letter-spacing: .12em;

    color: #8f96a8;
}

@keyframes rfLogoFloat {
    0%, 100% {
        transform: translateY(0) rotate(0deg);
    }

    50% {
        transform: translateY(-3px) rotate(2deg);
    }
}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.035);
        color: #eeeeef;
        font-weight: 600;
        min-height: 2.6rem;

        transition:
            transform .2s ease,
            background .2s ease,
            border-color .2s ease,
            box-shadow .2s ease;
    }


    .stButton > button:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,.065);
        border-color: rgba(155,135,245,.35);
        box-shadow: 0 12px 30px rgba(0,0,0,.25);
    }


    .stButton > button:active {
        transform: scale(.98);
    }


    /* ======================================================
       TEXT INPUT
       ====================================================== */

    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
        background: rgba(255,255,255,.035) !important;
        border: 1px solid rgba(255,255,255,.08) !important;
        color: white !important;
    }


    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(155,135,245,.5) !important;
        box-shadow: 0 0 0 3px rgba(155,135,245,.08) !important;
    }


    /* ======================================================
   STREAMLIT BOTTOM CONTAINER FIX
   ====================================================== */

[data-testid="stBottom"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}


/* ======================================================
   MODERN NEON CHAT INPUT
   ====================================================== */

[data-testid="stChatInput"] {
    position: relative !important;

    background: transparent !important;

    padding: 18px 0 38px 0 !important;

    margin-top: 12px !important;

    overflow: visible !important;
}


/* ======================================================
   MAIN GLASS INPUT CONTAINER
   ====================================================== */

[data-testid="stChatInput"] > div {
    position: relative !important;

    min-height: 62px !important;

    border-radius: 22px !important;

    border: 1px solid rgba(167,139,250,.55) !important;

    background:
        linear-gradient(
            135deg,
            rgba(30,25,65,.92),
            rgba(13,17,35,.94)
        ) !important;

    box-shadow:
        0 0 0 1px rgba(99,102,241,.10),
        0 0 25px rgba(139,92,246,.18),
        0 0 65px rgba(99,102,241,.10),
        inset 0 1px 0 rgba(255,255,255,.08) !important;

    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;

    overflow: visible !important;

    transition:
        border-color .25s ease,
        box-shadow .25s ease,
        transform .25s ease !important;
}


/* ======================================================
   INPUT HOVER
   ====================================================== */

[data-testid="stChatInput"] > div:hover {
    border-color: rgba(167,139,250,.78) !important;

    box-shadow:
        0 0 0 1px rgba(167,139,250,.14),
        0 0 30px rgba(139,92,246,.25),
        0 0 80px rgba(99,102,241,.12),
        inset 0 1px 0 rgba(255,255,255,.10) !important;
}


/* ======================================================
   INPUT FOCUS
   ====================================================== */

[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(167,139,250,.95) !important;

    transform: translateY(-2px) !important;

    box-shadow:
        0 0 0 1px rgba(167,139,250,.22),
        0 0 30px rgba(139,92,246,.35),
        0 0 90px rgba(99,102,241,.16),
        inset 0 1px 0 rgba(255,255,255,.12) !important;
}


/* ======================================================
   INPUT TEXT
   ====================================================== */

[data-testid="stChatInput"] textarea {
    background: transparent !important;

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    caret-color: #c4b5fd !important;

    font-size: 16px !important;

    font-weight: 500 !important;

    line-height: 1.5 !important;

    padding: 15px 58px 15px 18px !important;

    border: none !important;

    outline: none !important;

    box-shadow: none !important;
}


/* ======================================================
   PLACEHOLDER
   ====================================================== */

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,.62) !important;

    -webkit-text-fill-color: rgba(255,255,255,.62) !important;

    opacity: 1 !important;
}


/* ======================================================
   TEXTAREA FOCUS
   ====================================================== */

[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    outline: none !important;

    box-shadow: none !important;
}


/* ======================================================
   SEND BUTTON
   ====================================================== */

[data-testid="stChatInput"] button {
    width: 46px !important;

    height: 46px !important;

    min-width: 46px !important;

    min-height: 46px !important;

    border-radius: 50% !important;

    border: 1px solid rgba(255,255,255,.18) !important;

    background:
        linear-gradient(
            135deg,
            #a78bfa,
            #6366f1
        ) !important;

    color: #ffffff !important;

    box-shadow:
        0 0 18px rgba(139,92,246,.45),
        0 0 35px rgba(99,102,241,.22) !important;

    transition:
        transform .2s ease,
        box-shadow .2s ease !important;
}


/* ======================================================
   SEND BUTTON HOVER
   ====================================================== */

[data-testid="stChatInput"] button:hover {
    transform: scale(1.08) !important;

    background:
        linear-gradient(
            135deg,
            #c4b5fd,
            #818cf8
        ) !important;

    box-shadow:
        0 0 22px rgba(167,139,250,.65),
        0 0 45px rgba(99,102,241,.30) !important;
}


/* ======================================================
   SEND BUTTON ACTIVE
   ====================================================== */

[data-testid="stChatInput"] button:active {
    transform: scale(.94) !important;
}


/* ======================================================
   ANIMATED NEON WAVE
   ====================================================== */

[data-testid="stChatInput"] > div::after {
    content: "";

    position: absolute;

    left: -20px;
    right: -20px;

    bottom: -35px;

    height: 70px;

    pointer-events: none;

    z-index: -1;

    background:
        radial-gradient(
            ellipse at 20% 80%,
            rgba(124,58,237,.42),
            transparent 48%
        ),
        radial-gradient(
            ellipse at 55% 20%,
            rgba(99,102,241,.28),
            transparent 45%
        ),
        radial-gradient(
            ellipse at 85% 75%,
            rgba(168,85,247,.38),
            transparent 48%
        );

    filter: blur(14px);

    opacity: .85;

    animation:
        rfWaveGlow 6s ease-in-out infinite alternate;
}


/* ======================================================
   NEON WAVE LINE
   ====================================================== */

[data-testid="stChatInput"] > div::before {
    content: "";

    position: absolute;

    left: -30px;
    right: -30px;

    bottom: -22px;

    height: 35px;

    pointer-events: none;

    background:
        radial-gradient(
            ellipse at 15% 100%,
            transparent 48%,
            rgba(139,92,246,.75) 49%,
            transparent 51%
        ),
        radial-gradient(
            ellipse at 55% 0%,
            transparent 48%,
            rgba(99,102,241,.65) 49%,
            transparent 51%
        ),
        radial-gradient(
            ellipse at 90% 100%,
            transparent 48%,
            rgba(168,85,247,.70) 49%,
            transparent 51%
        );

    filter: blur(2px);

    opacity: .75;

    animation:
        rfWaveMove 7s ease-in-out infinite alternate;
}


/* ======================================================
   WAVE ANIMATION
   ====================================================== */

@keyframes rfWaveGlow {

    0% {
        transform:
            translateX(-25px)
            scaleX(.95);

        opacity: .55;
    }

    50% {
        transform:
            translateX(10px)
            scaleX(1.05);

        opacity: .85;
    }

    100% {
        transform:
            translateX(25px)
            scaleX(.98);

        opacity: .65;
    }
}


@keyframes rfWaveMove {

    0% {
        transform:
            translateX(-18px)
            scaleX(.95);
    }

    50% {
        transform:
            translateX(8px)
            scaleX(1.04);
    }

    100% {
        transform:
            translateX(20px)
            scaleX(.96);
    }
}


/* ======================================================
   RESPONSIVE
   ====================================================== */

@media (max-width: 768px) {

    [data-testid="stChatInput"] {
        padding-bottom: 32px !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 15px !important;

        padding-left: 14px !important;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 18px !important;
    }
}


/* ======================================================
   REDUCED MOTION
   ====================================================== */

@media (prefers-reduced-motion: reduce) {

    [data-testid="stChatInput"] > div::before,
    [data-testid="stChatInput"] > div::after {
        animation: none !important;
    }

}
  
    /* ======================================================
       CHAT
       ====================================================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        animation: messageAppear .4s ease both;
        margin-bottom: 1rem;
    }

    /* ======================================================
   RAGFURY ASSISTANT RESPONSE VISIBILITY
   ====================================================== */

/* Make the entire assistant response stand out */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,.055),
        rgba(155,135,245,.035)
    ) !important;

    border: 1px solid rgba(155,135,245,.18) !important;
    border-radius: 18px !important;

    padding: 16px !important;

    box-shadow:
        0 12px 40px rgba(0,0,0,.22),
        inset 0 1px 0 rgba(255,255,255,.045);

    backdrop-filter: blur(10px);
}


/* Make assistant answer text brighter and larger */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] {
    color: #f5f5f7 !important;
    font-size: 16px !important;
    line-height: 1.75 !important;
}


/* Make paragraphs inside RAGFury responses visible */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] p {
    color: #f5f5f7 !important;
    font-size: 16px !important;
    line-height: 1.75 !important;
}


/* Make headings inside responses brighter */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] h4 {
    color: #ffffff !important;
}


/* Make bold text stand out */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] strong {
    color: #ffffff !important;
}


/* RAGFury heading */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) h4 {
    color: #e9e2ff !important;
}


/* Improve markdown/code visibility */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) code {
    color: #e9e7ff !important;
}


    @keyframes messageAppear {

        from {
            opacity: 0;
            transform: translateY(12px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* ======================================================
       NATIVE CONTAINER CARDS
       ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border-color: rgba(255,255,255,.075) !important;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,.045),
                rgba(255,255,255,.018)
            );

        transition:
            transform .25s ease,
            border-color .25s ease,
            background .25s ease;
    }


    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(155,135,245,.18) !important;
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,.06),
                rgba(255,255,255,.025)
            );
    }


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.025);
        border: 1px solid rgba(255,255,255,.065);
        border-radius: 14px;
        padding: 12px;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        border-color: rgba(255,255,255,.075) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,.02);
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        border-color: rgba(255,255,255,.065) !important;
    }


    /* ======================================================
       STATUS
       ====================================================== */

    .rf-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 6px 10px;

        border-radius: 999px;

        background: rgba(255,255,255,.035);

        border: 1px solid rgba(255,255,255,.075);

        color: #aeb4c2;

        font-size: 10px;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }


    .rf-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--rf-green);

        box-shadow:
            0 0 10px rgba(66,217,155,.8);

        animation: pulseDot 2s infinite;
    }


    .rf-dot.offline {
        background: #fb7185;
        box-shadow:
            0 0 10px rgba(251,113,133,.8);
    }


    @keyframes pulseDot {

        0%,100% {
            transform: scale(.8);
            opacity: .6;
        }

        50% {
            transform: scale(1.2);
            opacity: 1;
        }
    }


    /* ======================================================
       HERO TITLE
       ====================================================== */

    .rf-title {
        font-size: clamp(42px, 6vw, 78px);
        line-height: .95;
        font-weight: 800;
        letter-spacing: -.065em;

        background:
            linear-gradient(
                110deg,
                #ffffff 15%,
                #c4b5fd 45%,
                #67e8f9 72%,
                #ffffff 95%
            );

        background-size: 250% auto;

        -webkit-background-clip: text;
        background-clip: text;

        color: transparent;

        animation: titleGradient 8s ease infinite;
    }


    @keyframes titleGradient {

        0%,100% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }
    }


    /* ======================================================
       KICKER
       ====================================================== */

    .rf-kicker {
        display: inline-block;

        padding: 7px 12px;

        border-radius: 999px;

        border: 1px solid rgba(155,135,245,.2);

        background: rgba(155,135,245,.045);

        color: #b8a8ff;

        font-size: 10px;

        letter-spacing: .13em;

        text-transform: uppercase;

        font-weight: 700;
    }


    /* ======================================================
       AGENT BADGE
       ====================================================== */

    .rf-agent {
        display: inline-flex;
        align-items: center;
        gap: 7px;

        margin-top: 7px;

        padding: 5px 9px;

        border-radius: 999px;

        background: rgba(255,255,255,.035);

        border: 1px solid rgba(255,255,255,.07);

        color: #9299a9;

        font-size: 9px;

        letter-spacing: .1em;

        font-weight: 700;
    }


    .rf-agent-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #a78bfa;

        box-shadow:
            0 0 9px rgba(167,139,250,.9);
    }


    /* ======================================================
       SOURCE
       ====================================================== */

    .rf-source {
        padding: 10px 13px;
        margin: 6px 0;

        border-radius: 11px;

        background: rgba(255,255,255,.025);

        border: 1px solid rgba(255,255,255,.06);

        transition:
            transform .2s ease,
            background .2s ease;
    }


    .rf-source:hover {
        transform: translateX(3px);
        background: rgba(255,255,255,.045);
    }


    .rf-source-title {
        font-size: 12px;
        font-weight: 600;
        color: #e1e3e8;
    }


    .rf-source-meta {
        font-size: 10px;
        color: #737b8d;
        margin-top: 3px;
    }


    /* ======================================================
       SECTION LABEL
       ====================================================== */

    .rf-label {
        font-size: 10px;
        font-weight: 700;
        color: #6f7687;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }


    /* ======================================================
       RESPONSIVE
       ====================================================== */

    @media(max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .rf-title {
            font-size: 48px;
        }

    }


    /* ======================================================
       REDUCED MOTION
       ====================================================== */

    @media(prefers-reduced-motion: reduce) {

        *,
        *::before,
        *::after {
            animation: none !important;
            transition: none !important;
        }

    }
    
    /* ======================================================
   RAGFURY — GLOBAL TEXT VISIBILITY
   ====================================================== */

/* Main application text */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #ffffff;
}


/* ======================================================
   HEADINGS
   ====================================================== */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #ffffff !important;
}


/* ======================================================
   MAIN "WHAT WOULD YOU LIKE TO KNOW?"
   ====================================================== */

.stApp h2 {
    color: #ffffff !important;
    font-weight: 700 !important;
}


/* ======================================================
   NORMAL DESCRIPTION / CAPTION TEXT
   ====================================================== */

.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: #f0f0f5 !important;
}


/* ======================================================
   SIDEBAR TEXT
   ====================================================== */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
}


/* ======================================================
   SIDEBAR SECTION LABELS
   ====================================================== */

section[data-testid="stSidebar"] .rf-label {
    color: #ffffff !important;
}


/* ======================================================
   MEMORY ARCHITECTURE
   RAGFURY ARCHITECTURE
   ====================================================== */

/* Expander header */
section[data-testid="stSidebar"]
[data-testid="stExpander"] summary,
section[data-testid="stSidebar"]
[data-testid="stExpander"] summary span {
    color: #ffffff !important;
}


/* Expander content */
section[data-testid="stSidebar"]
[data-testid="stExpander"] p,
section[data-testid="stSidebar"]
[data-testid="stExpander"] span,
section[data-testid="stSidebar"]
[data-testid="stExpander"] div {
    color: #ffffff !important;
}


/* ======================================================
   NEW CONVERSATION BUTTON
   ====================================================== */

section[data-testid="stSidebar"]
.stButton button {
    color: #ffffff !important;
}


/* ======================================================
   STATUS TEXT
   ====================================================== */

.rf-status {
    color: #ffffff !important;
}


/* ======================================================
   WORKSPACE INPUT
   ====================================================== */

section[data-testid="stSidebar"]
div[data-testid="stTextInput"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}


/* ======================================================
   CHAT INPUT
   ====================================================== */

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}


/* Chat placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #d6d9e2 !important;
    -webkit-text-fill-color: #d6d9e2 !important;
    opacity: 1 !important;
}


/* ======================================================
   CHAT / RAGFURY ANSWERS
   ====================================================== */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #ffffff !important;
}


[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #ffffff !important;
}


/* ======================================================
   CAPABILITY CARDS
   ====================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] span {
    color: #ffffff !important;
}


/* ======================================================
   SOURCE / CITATION TEXT
   ====================================================== */

.rf-source-title {
    color: #ffffff !important;
}

.rf-source-meta {
    color: #d0d4df !important;
}


/* ======================================================
   AGENT BADGE
   ====================================================== */

.rf-agent {
    color: #ffffff !important;
}


/* ======================================================
   MARKDOWN TEXT
   ====================================================== */

.stMarkdown p,
.stMarkdown li,
.stMarkdown strong,
.stMarkdown em {
    color: #ffffff !important;
}
    

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():

    if "history" not in st.session_state:
        st.session_state.history = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def handle_user_id_change():

    new_user_id = st.session_state.user_id_input.strip()

    if new_user_id == st.session_state.user_id:
        return

    st.session_state.user_id = new_user_id
    st.session_state.conversation_id = None
    st.session_state.history = []


def start_new_conversation():

    st.session_state.conversation_id = None
    st.session_state.history = []


# ============================================================
# API
# ============================================================

def check_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def ask_backend(question: str):

    payload = {
        "question": question,
        "user_id": st.session_state.user_id,
    }

    if st.session_state.conversation_id:

        payload["conversation_id"] = (
            st.session_state.conversation_id
        )

    response = requests.post(
        f"{API_URL}/api/v1/query",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    returned_conversation_id = result.get(
        "conversation_id"
    )

    if returned_conversation_id:

        st.session_state.conversation_id = (
            returned_conversation_id
        )

    return result


def submit_feedback(run_id, score):

    try:

        response = requests.post(
            f"{API_URL}/api/v1/feedback",
            json={
                "run_id": run_id,
                "score": score,
            },
            timeout=10,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:

        return False


# ============================================================
# 3D HERO COMPONENT
# ============================================================

def display_3d_visual():

    components.html(
        """
        <!DOCTYPE html>

        <html>

        <head>

        <style>

        html,
        body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: transparent;
        }


        .scene {

            width: 100%;
            height: 330px;

            position: relative;

            display: flex;
            align-items: center;
            justify-content: center;

            perspective: 900px;

        }


        .glow {

            position: absolute;

            width: 280px;
            height: 280px;

            border-radius: 50%;

            background:
                radial-gradient(
                    circle,
                    rgba(139,92,246,.20),
                    transparent 68%
                );

            filter: blur(20px);

            animation: glow 5s ease-in-out infinite;

        }


        .core {

            width: 145px;
            height: 145px;

            position: relative;

            border-radius: 50%;

            background:

                radial-gradient(
                    circle at 28% 24%,
                    rgba(255,255,255,.6),
                    transparent 8%
                ),

                radial-gradient(
                    circle at 38% 35%,
                    #b7a4ff,
                    #7149d8 35%,
                    #211d42 65%,
                    #07070d 100%
                );

            box-shadow:

                inset -25px -25px 45px
                rgba(0,0,0,.6),

                inset 12px 10px 25px
                rgba(255,255,255,.09),

                0 0 55px
                rgba(139,92,246,.35),

                0 0 130px
                rgba(139,92,246,.12);

            animation: float 5s ease-in-out infinite;

            z-index: 5;

        }


        .ring {

            position: absolute;

            width: 220px;
            height: 80px;

            border: 1px solid
                rgba(167,139,250,.35);

            border-radius: 50%;

            transform:
                rotateX(70deg)
                rotateZ(15deg);

            animation:
                spin 10s linear infinite;

        }


        .ring.two {

            width: 260px;
            height: 95px;

            border-color:
                rgba(103,232,249,.20);

            transform:
                rotateX(72deg)
                rotateY(15deg);

            animation-duration: 15s;

            animation-direction: reverse;

        }


        .ring.three {

            width: 190px;
            height: 190px;

            border-color:
                rgba(255,255,255,.07);

            transform:
                rotateY(70deg);

            animation-duration: 18s;

        }


        .particle {

            position: absolute;

            width: 4px;
            height: 4px;

            border-radius: 50%;

            background: #b7a4ff;

            box-shadow:
                0 0 12px #9b87f5;

        }


        .p1 {
            top: 30px;
            left: 120px;
            animation: particleOne 6s infinite;
        }


        .p2 {
            top: 210px;
            right: 100px;
            animation: particleTwo 7s infinite;
        }


        .p3 {
            bottom: 40px;
            left: 100px;
            animation: particleThree 8s infinite;
        }


        .p4 {
            top: 110px;
            right: 65px;
            animation: particleFour 9s infinite;
        }


        @keyframes float {

            0%,100% {
                transform:
                    translateY(0)
                    rotateY(0deg);
            }

            50% {
                transform:
                    translateY(-14px)
                    rotateY(12deg);
            }

        }


        @keyframes spin {

            from {
                transform:
                    rotateX(70deg)
                    rotateZ(0deg);
            }

            to {
                transform:
                    rotateX(70deg)
                    rotateZ(360deg);
            }

        }


        @keyframes glow {

            0%,100% {
                transform: scale(.9);
                opacity: .6;
            }

            50% {
                transform: scale(1.15);
                opacity: 1;
            }

        }


        @keyframes particleOne {

            0%,100% {
                transform: translate(0,0);
                opacity: .3;
            }

            50% {
                transform: translate(30px,-25px);
                opacity: 1;
            }

        }


        @keyframes particleTwo {

            0%,100% {
                transform: translate(0,0);
                opacity: .3;
            }

            50% {
                transform: translate(-25px,30px);
                opacity: 1;
            }

        }


        @keyframes particleThree {

            0%,100% {
                transform: translate(0,0);
                opacity: .3;
            }

            50% {
                transform: translate(35px,-20px);
                opacity: 1;
            }

        }


        @keyframes particleFour {

            0%,100% {
                transform: translate(0,0);
                opacity: .3;
            }

            50% {
                transform: translate(-30px,-25px);
                opacity: 1;
            }

        }

        </style>

        </head>

        <body>

            <div class="scene">

                <div class="glow"></div>

                <div class="ring"></div>

                <div class="ring two"></div>

                <div class="ring three"></div>

                <div class="core"></div>

                <div class="particle p1"></div>
                <div class="particle p2"></div>
                <div class="particle p3"></div>
                <div class="particle p4"></div>

            </div>

        </body>

        </html>
        """,
        height=350,
        scrolling=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

def display_sidebar(api_online):

    with st.sidebar:

        st.markdown(
    """<div class="rf-brand">
        <div class="rf-brand-icon">
            <span>✦</span>
        </div>
        <div class="rf-brand-text">
            <div class="rf-brand-name">RAG<span>Fury</span></div>
            <div class="rf-brand-subtitle">AGENTIC KNOWLEDGE INTELLIGENCE</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

        st.divider()

        st.markdown(
            '<div class="rf-label">Workspace</div>',
            unsafe_allow_html=True,
        )

        st.text_input(
            "EMP ID",
            value=st.session_state.user_id,
            placeholder="employee_12345",
            key="user_id_input",
            on_change=handle_user_id_change,
            label_visibility="collapsed",
        )

        if st.session_state.user_id:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot"></span>
                    Workspace active
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot offline"></span>
                    Workspace inactive
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="rf-label">Conversation</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "＋ New conversation",
            use_container_width=True,
        ):

            start_new_conversation()

            st.rerun()

        st.markdown(
            '<div class="rf-label">System</div>',
            unsafe_allow_html=True,
        )

        if api_online:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot"></span>
                    API online
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot offline"></span>
                    API offline
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander(
            "Memory architecture"
        ):

            st.markdown(
                """
                **Redis**

                Short-term conversation memory.

                **Mem0**

                Long-term semantic memory.

                **User ID**

                Long-term memory namespace.

                **Conversation ID**

                Current conversation.
                """
            )

        with st.expander(
            "RAGFury architecture"
        ):

            st.markdown(
                """
                **Agentic Routing**

                Selects the appropriate agent.

                **Hybrid Retrieval**

                Searches indexed knowledge.

                **Reranking**

                Improves contextual relevance.

                **Grounded Generation**

                Produces source-aware answers.
                """
            )


# ============================================================
# WELCOME SCREEN
# ============================================================

def display_welcome():

    left, right = st.columns(
        [1.15, .85],
        gap="large",
    )

    with left:

        st.markdown(
            '<div class="rf-kicker">✦ AGENTIC KNOWLEDGE INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="rf-title">
                Intelligence.<br>
                Grounded in<br>
                your knowledge.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            RAGFury automatically decides whether your
            request needs document research or natural
            conversation — then gives you a grounded,
            contextual answer.
            """,
        )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "##### 📚 Document Research"
            )

            st.caption(
                "Hybrid retrieval, reranking and grounded answers."
            )

        with c2:

            st.markdown(
                "##### ◌ Conversational AI"
            )

            st.caption(
                "Natural conversations with persistent context."
            )

    with right:

        display_3d_visual()

    st.divider()

    st.markdown(
        '<div class="rf-label">Capabilities</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(border=True):

            st.markdown(
                "### 📄"
            )

            st.markdown(
                "**Document Research**"
            )

            st.caption(
                "Search your indexed knowledge and receive grounded answers."
            )

    with c2:

        with st.container(border=True):

            st.markdown(
                "### 🧠"
            )

            st.markdown(
                "**Agentic Routing**"
            )

            st.caption(
                "RAGFury chooses the right reasoning path automatically."
            )

    with c3:

        with st.container(border=True):

            st.markdown(
                "### ◈"
            )

            st.markdown(
                "**Trust & Sources**"
            )

            st.caption(
                "Inspect citations and retrieved context when RAG is used."
            )


# ============================================================
# ROUTE
# ============================================================

def display_route(route):

    if route == "rag":

        text = "DOCUMENT RESEARCH AGENT"

    elif route == "chat":

        text = "CONVERSATIONAL AGENT"

    else:

        text = (
            f"AGENT · {route or 'UNKNOWN'}"
        )

    st.markdown(
        f"""
        <div class="rf-agent">
            <span class="rf-agent-dot"></span>
            {html.escape(text)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CITATIONS
# ============================================================

def display_citations(citations):

    if not citations:
        return

    st.markdown(
        '<div class="rf-label">Sources</div>',
        unsafe_allow_html=True,
    )

    for index, citation in enumerate(
        citations,
        start=1,
    ):

        citation_id = citation.get(
            "citation_id",
            f"S{index}",
        )

        source = html.escape(
            str(
                citation.get(
                    "source",
                    "Unknown source",
                )
            )
        )

        page = citation.get("page")

        chunk_id = citation.get(
            "chunk_id",
            "unknown",
        )

        metadata = []

        if page is not None:

            metadata.append(
                f"Page {page}"
            )

        if chunk_id:

            metadata.append(
                f"Chunk {html.escape(str(chunk_id))}"
            )

        st.markdown(
            f"""
            <div class="rf-source">

                <div class="rf-source-title">
                    {html.escape(str(citation_id))}
                    ·
                    {source}
                </div>

                <div class="rf-source-meta">
                    {" · ".join(metadata)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RAG DETAILS
# ============================================================

def display_rag_details(result):

    if result.get("next_step") != "rag":
        return

    documents = result.get(
        "documents",
        [],
    )

    document_relevance = result.get(
        "document_relevance"
    )

    grade_reason = result.get(
        "grade_reason"
    )

    retrieval_attempts = result.get(
        "retrieval_attempts"
    )

    if not (
        documents
        or document_relevance is not None
        or grade_reason
        or retrieval_attempts is not None
    ):
        return

    with st.expander(
        "Research intelligence"
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Retrieval",
                (
                    retrieval_attempts
                    if retrieval_attempts is not None
                    else "—"
                ),
            )

        with c2:

            st.metric(
                "Documents",
                len(documents),
            )

        with c3:

            st.metric(
                "Relevance",
                (
                    document_relevance
                    if document_relevance is not None
                    else "—"
                ),
            )

        if grade_reason:

            st.markdown(
                "**Grading reason**"
            )

            st.caption(
                grade_reason
            )

        if documents:

            st.markdown(
                "**Retrieved context**"
            )

            for index, document in enumerate(
                documents,
                start=1,
            ):

                content = document.get(
                    "content",
                    "",
                )

                metadata = document.get(
                    "metadata",
                    {},
                )

                with st.expander(
                    f"Document {index}"
                ):

                    if content:

                        st.write(
                            content
                        )

                    if metadata:

                        st.caption(
                            f"Metadata: {metadata}"
                        )


# ============================================================
# FEEDBACK
# ============================================================

def display_feedback(run_id):

    if not run_id:
        return

    st.caption(
        "Was this response useful?"
    )

    c1, c2, _ = st.columns(
        [1, 1, 6]
    )

    with c1:

        if st.button(
            "👍 Helpful",
            key=f"positive_{run_id}",
        ):

            if submit_feedback(
                run_id,
                1.0,
            ):

                st.toast(
                    "Thanks for the feedback!"
                )

    with c2:

        if st.button(
            "👎 Not helpful",
            key=f"negative_{run_id}",
        ):

            if submit_feedback(
                run_id,
                0.0,
            ):

                st.toast(
                    "Thanks for the feedback!"
                )


# ============================================================
# HISTORY
# ============================================================

def display_history():

    for item in st.session_state.history:

        with st.chat_message(
            "user"
        ):

            st.write(
                item["question"]
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                "#### ✦ RAGFury"
            )

            st.write(
                item["answer"]
            )

            display_route(
                item.get(
                    "route",
                    "unknown",
                )
            )

            if item.get(
                "route"
            ) == "rag":

                display_citations(
                    item.get(
                        "citations",
                        [],
                    )
                )

                stored_result = item.get(
                    "result"
                )

                if stored_result:

                    display_rag_details(
                        stored_result
                    )

            display_feedback(
                item.get(
                    "run_id"
                )
            )

            backend_time = item.get(
                "response_time"
            )

            ui_time = item.get(
                "time"
            )

            if backend_time is not None:

                st.caption(
                    f"{backend_time:.2f}s backend · "
                    f"{ui_time:.2f}s total"
                )

            elif ui_time is not None:

                st.caption(
                    f"{ui_time:.2f}s total"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    init_session_state()

    api_online = check_api_health()

    display_sidebar(
        api_online
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    left, right = st.columns(
        [8, 2]
    )

    with left:

        st.caption(
            "RAGFURY / AGENTIC KNOWLEDGE WORKSPACE"
        )

    with right:

        if api_online:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot"></span>
                    API ONLINE
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="rf-status">
                    <span class="rf-dot offline"></span>
                    API OFFLINE
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # OFFLINE
    # --------------------------------------------------------

    if not api_online:

        st.markdown(
            "# Backend unavailable"
        )

        st.warning(
            "RAGFury API is currently unavailable."
        )

        st.code(
            "uvicorn api.main:app --reload",
            language="powershell",
        )

        return

    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    if not st.session_state.user_id:

        display_welcome()

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    elif st.session_state.history:

        display_history()

    # --------------------------------------------------------
    # EMPTY USER WORKSPACE
    # --------------------------------------------------------

    else:

        display_3d_visual()

        st.markdown(
            """
            <div style="
                text-align:center;
                margin-top:-20px;
            ">
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "## What would you like to know?"
        )

        st.caption(
            "Ask about your documents or start a natural conversation."
        )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask RAGFury anything..."
    )

    if question is None:
        return

    # --------------------------------------------------------
    # USER VALIDATION
    # --------------------------------------------------------

    if not st.session_state.user_id:

        st.error(
            "Please enter your EMP ID first."
        )

        return

    question = question.strip()

    if not question:

        return

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )

    # --------------------------------------------------------
    # THINKING
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            """
            <div class="rf-status">
                <span class="rf-dot"></span>
                RAGFURY IS THINKING
            </div>
            """,
            unsafe_allow_html=True,
        )

    start_time = time.time()

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    try:

        result = ask_backend(
            question
        )

        elapsed_time = (
            time.time() - start_time
        )

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        route = result.get(
            "next_step",
            "unknown",
        )

        # ----------------------------------------------------
        # SAVE TURN
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "question": question,
                "answer": answer,
                "time": elapsed_time,
                "route": route,
                "run_id": result.get(
                    "run_id"
                ),
                "citations": result.get(
                    "citations",
                    [],
                ),
                "documents": result.get(
                    "documents",
                    [],
                ),
                "document_relevance": result.get(
                    "document_relevance"
                ),
                "grade_reason": result.get(
                    "grade_reason"
                ),
                "retrieval_attempts": result.get(
                    "retrieval_attempts"
                ),
                "response_time": result.get(
                    "response_time"
                ),
                "result": result,
            }
        )

        st.rerun()

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The RAGFury pipeline timed out."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "🔴 Could not connect to the FastAPI backend."
        )

    except requests.exceptions.HTTPError as exc:

        try:

            error_data = (
                exc.response.json()
            )

            detail = error_data.get(
                "detail",
                "The request was rejected by the API.",
            )

        except Exception:

            detail = (
                "The request was rejected by the API."
            )

        if exc.response.status_code == 400:

            st.warning(
                f"🛡️ {detail}"
            )

        elif exc.response.status_code == 503:

            st.error(
                f"🚨 {detail}"
            )

        else:

            st.error(
                f"❌ {detail}"
            )

    except Exception as exc:

        st.error(
            f"❌ Unexpected error: {exc}"
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
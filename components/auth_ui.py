"""3D animated glassmorphic Sign In / Sign Up component supporting email & phone auth."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from auth import (
    authenticate_with_password,
    get_or_create_demo_user,
    is_smtp_configured,
    register_user,
    request_otp,
    verify_otp_login,
)
from components.spatial_3d import render_spatial_canvas
from database import get_db_target_info


def _complete_login(user: dict[str, Any], message: str) -> None:
    st.session_state["auth_user"] = user
    st.session_state.pop("demo_otp_preview", None)
    st.session_state.pop("ai_insights_markdown", None)
    st.session_state.pop("ai_insights_fingerprint", None)
    st.session_state["last_action_toast"] = message
    st.rerun()


def render_auth_portal() -> Optional[dict[str, Any]]:
    """Render the centered 3D animated Sign In / Create Account portal if unauthenticated."""
    existing_user = st.session_state.get("auth_user")
    if isinstance(existing_user, dict) and existing_user.get("id"):
        return existing_user

    render_spatial_canvas(mode="auth")

    db_info = get_db_target_info()
    db_badge_color = "#34d399" if db_info.get("is_cloud") else "#60a5fa"

    st.markdown(
        f"""
        <div class="auth-hero-wrap">
            <div class="cube-stage" aria-hidden="true">
                <div class="cube-3d">
                    <div class="cube-face cube-front">₹</div>
                    <div class="cube-face cube-back">$</div>
                    <div class="cube-face cube-right">AI</div>
                    <div class="cube-face cube-left">₿</div>
                    <div class="cube-face cube-top">▲</div>
                    <div class="cube-face cube-bottom">◆</div>
                </div>
            </div>
            <div class="auth-hero-title">AI-Powered Finance &amp; Investment Vault</div>
            <div class="auth-hero-sub">
                Sign in with your <b>Email</b> or <b>Phone Number</b> to access your 3D financial command center.
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px;">
                Storage Engine: <span style="color:{db_badge_color}; font-weight:600;">{db_info.get('display', '')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.45, 1])
    with center_col:
        with st.container(border=True):
            tab_signin, tab_signup = st.tabs(["Sign In", "Create Account"])

            with tab_signin:
                login_mode = st.radio(
                    "Authentication Method",
                    options=["Email / Phone + Password", "Email / Phone + 6-Digit OTP"],
                    horizontal=True,
                    key="auth_login_mode",
                )

                if login_mode == "Email / Phone + Password":
                    with st.form("auth_password_login_form", clear_on_submit=False):
                        identifier = st.text_input(
                            "Email or Phone Number",
                            placeholder="alex@example.com or +919876543210",
                            key="login_identifier_input",
                        )
                        password = st.text_input(
                            "Password",
                            type="password",
                            placeholder="Enter your password",
                            key="login_password_input",
                        )
                        submit_pw = st.form_submit_button(
                            "Sign In",
                            type="primary",
                            width="stretch",
                        )
                    if submit_pw:
                        try:
                            user = authenticate_with_password(identifier, password)
                            _complete_login(user, f"Welcome back, {user['full_name']}!")
                        except ValueError as exc:
                            st.error(str(exc))

                else:
                    otp_identifier = st.text_input(
                        "Registered Email or Phone Number",
                        placeholder="alex@example.com or +919876543210",
                        key="otp_login_identifier",
                    )
                    if st.button(
                        "Send 6-Digit OTP",
                        key="send_otp_btn",
                        width="stretch",
                    ):
                        try:
                            code, target_user = request_otp(otp_identifier)
                            st.session_state["demo_otp_preview"] = {
                                "identifier": otp_identifier.strip(),
                                "code": code,
                                "name": target_user["full_name"],
                            }
                        except ValueError as exc:
                            st.error(str(exc))

                    preview = st.session_state.get("demo_otp_preview")
                    if isinstance(preview, dict):
                        if is_smtp_configured() and "@" in str(preview.get("identifier", "")):
                            st.success(
                                f"📧 6-Digit OTP code dispatched directly to **{preview['identifier']}**! Check your email inbox."
                            )
                        else:
                            st.info(
                                f"Local Demo OTP for **{preview['name']}**: `{preview['code']}` "
                                "(valid for 5 minutes; configure SMTP in `.env` for direct email delivery)."
                            )

                    with st.form("auth_otp_verify_form", clear_on_submit=False):
                        otp_code = st.text_input(
                            "6-Digit OTP Code",
                            max_chars=6,
                            placeholder="Enter 6-digit code",
                            key="otp_code_input",
                        )
                        submit_otp = st.form_submit_button(
                            "Verify & Sign In",
                            type="primary",
                            width="stretch",
                        )
                    if submit_otp:
                        try:
                            user = verify_otp_login(otp_identifier, otp_code)
                            _complete_login(user, f"OTP verified — welcome, {user['full_name']}!")
                        except ValueError as exc:
                            st.error(str(exc))

            with tab_signup:
                with st.form("auth_signup_form", clear_on_submit=False):
                    full_name = st.text_input(
                        "Full Name",
                        placeholder="e.g., Aarav Sharma",
                        key="signup_name_input",
                    )
                    primary_contact = st.text_input(
                        "Email or Phone Number *",
                        placeholder="aarav@example.com or +919876543210",
                        key="signup_primary_input",
                    )
                    secondary_contact = st.text_input(
                        "Backup Phone or Email (Optional)",
                        placeholder="Optional alternate sign-in contact",
                        key="signup_secondary_input",
                    )
                    pw1 = st.text_input(
                        "Password (min 6 chars) *",
                        type="password",
                        key="signup_pw1_input",
                    )
                    pw2 = st.text_input(
                        "Confirm Password *",
                        type="password",
                        key="signup_pw2_input",
                    )
                    submit_signup = st.form_submit_button(
                        "Create Account & Sign In",
                        type="primary",
                        width="stretch",
                    )

                if submit_signup:
                    if pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            new_user = register_user(
                                full_name=full_name,
                                identifier=primary_contact,
                                password=pw1,
                                secondary_identifier=secondary_contact or None,
                            )
                            _complete_login(
                                new_user,
                                f"Account created — welcome, {new_user['full_name']}!",
                            )
                        except ValueError as exc:
                            st.error(str(exc))

        st.write("")
        if st.button(
            "Launch Instant Demo Account (Preloaded Portfolio)",
            key="auth_demo_login_btn",
            width="stretch",
        ):
            demo_user = get_or_create_demo_user()
            _complete_login(demo_user, "Signed in to preloaded Demo Account.")

    return None


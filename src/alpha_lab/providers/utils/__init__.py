"""Vendor-agnostic connectivity for the providers package.

Auth (bearer token, on-prem detection, MS CA bundle) and the shared on-prem
client transport. This subpackage is a leaf: it imports nothing from the
provider modules above it. Kept intentionally thin (no eager re-exports) so the
facade -> factory -> provider -> core import chain can't trip a partial-init
cycle; import the submodules directly.
"""

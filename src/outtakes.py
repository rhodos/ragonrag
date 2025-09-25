

# # normalized_vertices can be different types depending on the client/runtime:
# # - protobuf message with to_dict()
# # - an object with .x and .y attributes
# # - a plain mapping/dict
# def _vertex_to_dict(v):
#     try:
#         # some client objects expose to_dict()
#         if hasattr(v, "to_dict"):
#             logger.debug('Using to_dict() for vertex')
#             return v.to_dict()
#     except Exception:
#         pass
#     # proto-like objects often have .x and .y
#     if hasattr(v, "x") and hasattr(v, "y"):
#         try:
#             logger.debug('Using .x and .y attributes for vertex')
#             return {"x": float(v.x), "y": float(v.y)}
#         except Exception:
#             return {"x": v.x, "y": v.y}
#     # mapping-like
#     if hasattr(v, "get"):
#         x = v.get("x")
#         y = v.get("y")
#         logger.debug('Using .get() for vertex')
#         if x is not None and y is not None:
#             return {"x": float(x), "y": float(y)}
#     # last resort: try to use __dict__ or string
#     try:
#         d = dict(v.__dict__)
#         if "x" in d and "y" in d:
#             return {"x": d.get("x"), "y": d.get("y")}
#     except Exception:
#         pass
#     # Give up and return a string representation to avoid crashes
#     return {"x": None, "y": None, "raw": str(v)}

# if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
#     logger.warning(
#         "GOOGLE_APPLICATION_CREDENTIALS not set in environment. "
#         "Document AI client will fail if credentials aren't provided via other means."
#     )

# if not os.environ.get("DOCAI_PROCESSOR_NAME"):
#     logger.warning(
#         "DOCAI_PROCESSOR_NAME not set in environment. "
#         "Document AI client will fail if processor name isn't provided via other means."
#     )

#print("\n\n".join([b.text for b in blocks[idx:]]))
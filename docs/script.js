const tabRoot = document.querySelector("[data-tabs]");

if (tabRoot) {
  const tabs = [...tabRoot.querySelectorAll("[role='tab']")];
  const panels = [...tabRoot.querySelectorAll("[role='tabpanel']")];

  const activateTab = (tab) => {
    const target = tab.dataset.tab;

    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    });

    panels.forEach((panel) => {
      const selected = panel.dataset.panel === target;
      panel.hidden = !selected;
      panel.classList.toggle("is-active", selected);
    });
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const nextIndex = (index + direction + tabs.length) % tabs.length;
      tabs[nextIndex].focus();
      activateTab(tabs[nextIndex]);
    });
  });
}

const citation = document.querySelector("#bibtex code")?.textContent.trim();
const toast = document.querySelector("[data-toast]");
let toastTimer;

const showToast = (message) => {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
};

const selectCitation = () => {
  const code = document.querySelector("#bibtex code");
  if (!code) return;
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(code);
  selection.removeAllRanges();
  selection.addRange(range);
};

const copyText = async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the selection fallback when permission is unavailable.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) selectCitation();
  return copied;
};

document.querySelectorAll("[data-copy-citation]").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!citation) return;
    try {
      const copied = await copyText(citation);
      showToast(copied ? "BibTeX copied" : "BibTeX selected");
    } catch {
      selectCitation();
      showToast("BibTeX selected");
    }
  });
});

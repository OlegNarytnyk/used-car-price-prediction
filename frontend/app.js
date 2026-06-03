const form = document.getElementById("prediction-form");
const brandSelect = document.getElementById("brand");
const modelSelect = document.getElementById("model");
const transmissionSelect = document.getElementById("transmission");
const fuelTypeSelect = document.getElementById("fuelType");
const loading = document.getElementById("loading");
const errorMessage = document.getElementById("error-message");
const result = document.getElementById("result");
const predictButton = document.getElementById("predict-button");

const fallbackTransmissions = ["Manual", "Automatic", "Semi-Auto"];
const fallbackFuelTypes = ["Petrol", "Diesel", "Hybrid"];

function setLoading(isLoading) {
  loading.hidden = !isLoading;
  predictButton.disabled = isLoading;
}

function showError(message) {
  errorMessage.textContent = message;
  result.textContent = "";
}

function clearMessages() {
  errorMessage.textContent = "";
  result.textContent = "";
}

function populateSelect(selectElement, values, preferredValue) {
  selectElement.innerHTML = "";

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectElement.appendChild(option);
  });

  if (preferredValue && values.includes(preferredValue)) {
    selectElement.value = preferredValue;
  }
}

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return data;
}

async function loadMetadata() {
  try {
    const metadata = await fetchJson("/metadata");
    populateSelect(transmissionSelect, metadata.transmissions || fallbackTransmissions, "Manual");
    populateSelect(fuelTypeSelect, metadata.fuel_types || fallbackFuelTypes, "Petrol");
  } catch (error) {
    populateSelect(transmissionSelect, fallbackTransmissions, "Manual");
    populateSelect(fuelTypeSelect, fallbackFuelTypes, "Petrol");
  }
}

async function loadBrands() {
  const brands = await fetchJson("/brands");
  populateSelect(brandSelect, brands, brands.includes("Ford") ? "Ford" : brands[0]);
}

async function loadModels(brand) {
  modelSelect.disabled = true;
  modelSelect.innerHTML = "";

  const models = await fetchJson(`/models/${encodeURIComponent(brand)}`);
  populateSelect(modelSelect, models, models.includes("Fiesta") ? "Fiesta" : models[0]);
  modelSelect.disabled = false;
}

function getNumber(name) {
  const value = Number(form.elements[name].value);

  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a valid number`);
  }

  return value;
}

function validatePayload(payload) {
  const requiredFields = ["brand", "model", "transmission", "fuelType"];

  requiredFields.forEach((field) => {
    if (!payload[field] || payload[field].trim() === "") {
      throw new Error(`${field} is required`);
    }
  });

  if (payload.year < 1990 || payload.year > 2026) {
    throw new Error("Year must be between 1990 and 2026");
  }

  if (payload.mileage < 0) {
    throw new Error("Mileage must be greater than or equal to 0");
  }

  if (payload.tax < 0) {
    throw new Error("Tax must be greater than or equal to 0");
  }

  if (payload.mpg <= 0) {
    throw new Error("MPG must be greater than 0");
  }

  if (payload.engineSize <= 0) {
    throw new Error("Engine Size must be greater than 0");
  }
}

function buildPayload() {
  const payload = {
    brand: form.elements.brand.value,
    model: form.elements.model.value,
    year: getNumber("year"),
    transmission: form.elements.transmission.value,
    mileage: getNumber("mileage"),
    fuelType: form.elements.fuelType.value,
    tax: getNumber("tax"),
    mpg: getNumber("mpg"),
    engineSize: getNumber("engineSize"),
  };

  payload.year = Math.trunc(payload.year);
  payload.mileage = Math.trunc(payload.mileage);
  validatePayload(payload);
  return payload;
}

async function predictPrice(event) {
  event.preventDefault();
  clearMessages();

  let payload;

  try {
    payload = buildPayload();
  } catch (error) {
    showError(error.message);
    return;
  }

  setLoading(true);

  try {
    const prediction = await fetchJson("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const formattedPrice = new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency: "GBP",
      maximumFractionDigits: 0,
    }).format(prediction.predicted_price);

    result.textContent = `Predicted price: ${formattedPrice}`;
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
}

async function initialize() {
  setLoading(true);

  try {
    await loadMetadata();
    await loadBrands();
    await loadModels(brandSelect.value);
  } catch (error) {
    showError(`Could not load API data: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

brandSelect.addEventListener("change", async () => {
  clearMessages();
  setLoading(true);

  try {
    await loadModels(brandSelect.value);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
});

form.addEventListener("submit", predictPrice);
initialize();
